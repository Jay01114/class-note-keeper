# -*- coding: utf-8 -*-
"""课堂笔记管家 · 核心流水线（M1 命令行版）
流程: 监视 inbox -> 新音频入队 -> faster-whisper 转写 -> Qwen 纪要+识别学科课名 -> 写入 Obsidian vault
用法: python core.py  （启动后常驻监视，Ctrl+C 退出；启动时先处理 inbox 已有文件）
"""
import os
import sys
import json
import time
import hashlib
import re
import shutil
from pathlib import Path
from datetime import datetime

if sys.stdout:
    sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内模型镜像
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 镜像站不支持 xet，走普通 HTTP

import requests
import json5
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

CONFIG_PATH = None
import os as _os
_env_cfg = _os.environ.get("KETANG_CONFIG", "")
for _cand in [
    Path(_env_cfg) if _env_cfg else None,             # 环境变量指定（最高优先）
    Path.cwd() / "config.json",                       # 当前目录
    Path(__file__).parent / "config.json",            # 脚本同级
    Path(__file__).parent.parent / "config.json",     # 上级
]:
    if _cand and _cand.exists():
        CONFIG_PATH = _cand
        break
if CONFIG_PATH is None:
    CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_config()

# 全局停止标志（GUI 调用 request_stop 停止轮询）
STOP_FLAG = False


def request_stop():
    global STOP_FLAG
    STOP_FLAG = True

# ---------------- 转写 ----------------
_model = None


def get_whisper():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"[转写] 加载模型 {CFG['whisper_model']} ({CFG['whisper_compute']}) ...")
        _model = WhisperModel(
            CFG["whisper_model"],
            device="cuda",
            compute_type=CFG["whisper_compute"],
            download_root=CFG["models_dir"],
        )
    return _model


def transcribe(path: str) -> str:
    model = get_whisper()
    name = Path(path).name
    print(f"[转写] 开始: {name}")
    t0 = time.time()
    segments, info = model.transcribe(path, language="zh", vad_filter=False)
    lines = []
    for seg in segments:
        h, m, s = int(seg.start // 3600), int(seg.start % 3600 // 60), int(seg.start % 60)
        lines.append(f"[{h:02d}:{m:02d}:{s:02d}] {seg.text.strip()}")
    text = "\n".join(lines)
    print(f"[转写] 完成，用时 {time.time() - t0:.1f}s，语言 {info.language} 置信度 {info.language_probability:.2f}")
    return text


# ---------------- 纪要 + 学科识别 ----------------
def ollama_chat(prompt: str) -> str:
    url = CFG["ollama_host"] + "/api/chat"
    payload = {
        "model": CFG["ollama_model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": 24576, "temperature": 0.2},
    }
    r = requests.post(url, json=payload, timeout=900)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _classify(full_text: str) -> dict:
    """第一步：识别学科/课程/标题（短 prompt，格式简单易遵守）"""
    excerpt = full_text[:12000]
    subject_list = "、".join(CFG.get("subjects", ["待整理"]))
    prompt = f"""判断下面课堂录音转写内容属于哪个学科、哪门课、什么标题。
学科只能从下面列表选一个，不要自创：
{subject_list}
注意：课程介绍、绪论、考核说明这类内容，如果明显是某门课的开课介绍，也归入该学科；只有与以上学科完全无关的内容（如外语课、闲聊）才判"待整理"。

只输出三行，不要任何其他文字：
SUBJECT: 学科
COURSE: 课程名
TITLE: 本节课标题

转写内容（开头部分）：
{excerpt}"""
    raw = ollama_chat(prompt)
    m_s = re.search(r"SUBJECT:\s*(.+)", raw)
    m_c = re.search(r"COURSE:\s*(.+)", raw)
    m_t = re.search(r"TITLE:\s*(.+)", raw)
    subject = normalize_subject(m_s.group(1).strip() if m_s else "")
    return {
        "subject": subject,
        "course": (m_c.group(1).strip() if m_c else subject),
        "title": (m_t.group(1).strip() if m_t else "未命名课程"),
    }


def _summarize(full_text: str, meta: dict) -> str:
    """第二步：生成详细知识点罗列正文"""
    excerpt = full_text[:20000]
    prompt = f"""把下面课堂录音转写文本整理成详细的知识点罗列笔记。
学科：{meta['subject']}｜课程：{meta['course']}｜标题：{meta['title']}

要求：
- 把课堂内容逐条整理成知识点，宁全勿简：定义、概念、公式、定律、性质、例子、数据、要求、注意事项都尽量保留；
- 删除口语废话和课堂套话（"同学们好""我们来看""那么""好的""下课"等寒暄连接词），但知识点信息本身要完整；
- 按主题分小节（## 开头），每条知识点用 - 开头；
- 直接从第一个小节开始组织，不要先输出总览/要点列表再重复展开，每个知识点只出现一次；
- 不要写成概括性总结，不要写"本节课主要讲述了…"这类综述，不要写客套结束语。

只输出正文，正文放在下面两个标记之间，不要输出其他内容：
SUMMARY_START
（正文）
SUMMARY_END

转写文本：
{excerpt}"""
    raw = ollama_chat(prompt)
    m = re.search(r"SUMMARY_START\s*(.*?)\s*SUMMARY_END", raw, re.S)
    return m.group(1).strip() if m else raw.strip()[:8000]


def generate_note(full_text: str) -> dict:
    print(f"[纪要] 识别学科/课程 ...")
    meta = _classify(full_text)
    print(f"[纪要] 学科={meta['subject']} 课程={meta['course']} 标题={meta['title']}，生成知识点 ...")
    t0 = time.time()
    summary = _summarize(full_text, meta)
    print(f"[纪要] 完成，用时 {time.time() - t0:.0f}s")
    return {**meta, "summary_md": summary}


# ---------------- 归档到 Obsidian vault ----------------
def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "未命名"


def normalize_subject(name: str) -> str:
    """把 AI 输出的学科名归一到已知学科列表（支持简称/变体，如"信号和系统"→"电路、信号和系统"）"""
    known = CFG.get("subjects", [])
    name = (name or "").strip()
    if name in known:
        return name
    for k in known:
        if name and (name in k or k in name):
            return k
    return "待整理"


def archive(path: Path, text: str, note: dict, file_hash: str = "") -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    # 学科归一化：只允许已知学科，未知的进"待整理"（不新建学科目录）
    subject = sanitize(normalize_subject(note.get("subject")))
    title = sanitize(note.get("title") or path.stem)
    course = sanitize(note.get("course") or title)

    # 结构：学科/{原材料,纪要,转写}/ 直接放文件（文件名 = 日期_标题）
    vault = Path(CFG["vault_dir"])
    lesson = f"{date_str}_{title}"
    raw_dir = vault / subject / "原材料"
    note_dir = vault / subject / "纪要"
    trans_dir = vault / subject / "转写"
    for d in (raw_dir, note_dir, trans_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 音频 → 原材料/日期_标题.ext（同名加时间戳后缀，保证从 inbox 移走）
    audio_dst = raw_dir / f"{lesson}{path.suffix}"
    if audio_dst.exists():
        audio_dst = raw_dir / f"{lesson}_{int(time.time())}{path.suffix}"
    shutil.move(str(path), str(audio_dst))

    # 转写 → 转写/日期_标题.md
    (trans_dir / f"{lesson}.md").write_text(
        f"# {title}\n\n学科：{subject}｜课程：{course}｜日期：{date_str}\n\n{text}\n",
        encoding="utf-8",
    )

    # 纪要 → 纪要/日期_标题.md
    frontmatter = (
        f"---\n"
        f"title: {title}\n"
        f"subject: {subject}\n"
        f"course: {course}\n"
        f"date: {date_str}\n"
        f"tags: [课堂笔记, {subject}]\n"
        f"---\n\n"
    )
    (note_dir / f"{lesson}.md").write_text(
        frontmatter + (note.get("summary_md") or ""), encoding="utf-8"
    )

    meta = {
        "file": audio_dst.name,
        "subject": subject,
        "course": course,
        "title": title,
        "date": date_str,
        "file_hash": file_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (raw_dir / f"{lesson}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[归档] -> {vault / subject}（原材料/纪要/转写）")
    return vault / subject


# ---------------- 去重 ----------------
def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_processed_hashes() -> set:
    """扫描 vault 中所有 meta.json 的 file_hash，构建已处理集合（用于去重）"""
    hashes = set()
    vault = Path(CFG["vault_dir"])
    if vault.exists():
        for meta in vault.rglob("*.meta.json"):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                if data.get("file_hash"):
                    hashes.add(data["file_hash"])
            except Exception:
                pass
    return hashes


PROCESSED_HASHES = set()


def move_to(path: Path, subdir_name: str) -> Path:
    """把文件移到 inbox 的子目录（_重复 / _失败），避免反复轮询"""
    sub = Path(CFG["inbox_dir"]) / subdir_name
    sub.mkdir(parents=True, exist_ok=True)
    dst = sub / path.name
    if dst.exists():
        dst = sub / f"{path.stem}_{int(time.time())}{path.suffix}"
    shutil.move(str(path), str(dst))
    return dst


# ---------------- 监视（轮询模式，兼容一切导入方式） ----------------
def process_one(path: str):
    p = Path(path)
    print(f"\n===== 处理: {p.name} =====")
    try:
        # 哈希去重：与 vault 已归档内容相同则直接移出，不重复整理
        h = file_hash(p)
        if h in PROCESSED_HASHES:
            dst = move_to(p, "_重复")
            print(f"[跳过] 内容已处理过（{p.name}）-> {dst}")
            return
        text = transcribe(str(p))
        note = generate_note(text)
        archive(p, text, note, file_hash=h)
        PROCESSED_HASHES.add(h)
        print(f"===== 完成: {p.name} =====\n")
    except Exception as e:
        print(f"[失败] {p.name}: {e}")
        try:
            dst = move_to(p, "_失败")
            print(f"[已移出] 避免重复处理 -> {dst}\n")
        except Exception:
            pass


# 忽略的临时/传输中后缀（LocalSend 等传输产生的中间文件不处理）
IGNORED_SUFFIXES = {".part", ".tmp", ".crdownload", ".download", ".partial", ".!qB"}

# 文件稳定时间（秒）：大小连续 N 秒不变才认为传输完成
STABLE_SECONDS = 6


def is_ready_audio(f: Path, stable: dict, now: float) -> bool:
    """文件完整性检测：扩展名合法 + 大小连续 STABLE_SECONDS 秒不变才就绪。
    stable: {name: (size, first_seen_time)}"""
    if not f.is_file():
        return False
    if f.suffix.lower() in IGNORED_SUFFIXES:
        return False
    if f.suffix.lower() not in CFG["supported_ext"]:
        return False
    size = f.stat().st_size
    prev = stable.get(f.name)
    if prev is None:
        stable[f.name] = (size, now)
        return False
    if prev[0] != size:
        stable[f.name] = (size, now)  # 大小还在变 → 传输中
        return False
    return now - prev[1] >= STABLE_SECONDS


def main_loop():
    global PROCESSED_HASHES
    inbox = Path(CFG["inbox_dir"])
    inbox.mkdir(parents=True, exist_ok=True)
    Path(CFG["vault_dir"]).mkdir(parents=True, exist_ok=True)
    PROCESSED_HASHES = load_processed_hashes()
    print(f"[去重] 已加载 {len(PROCESSED_HASHES)} 条历史记录")

    print(f"[监视] 轮询 {inbox}，每 3 秒扫一次（关闭窗口即停止）")
    seen = set()  # 只记录已开始处理的文件；启动时已有的文件走稳定检测
    stable: dict = {}
    while not STOP_FLAG:
        try:
            now = time.time()
            for f in sorted(inbox.iterdir()):
                if f.name in seen:
                    continue
                if is_ready_audio(f, stable, now):
                    seen.add(f.name)
                    process_one(str(f))
            time.sleep(3)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[监视异常] {e}")
            time.sleep(5)


if __name__ == "__main__":
    main_loop()
