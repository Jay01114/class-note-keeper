# -*- coding: utf-8 -*-
"""课堂笔记管家 · 核心流水线
流程: 监视 inbox -> 新音频入队 -> faster-whisper 转写 -> Qwen 纪要+识别学科课名 -> 写入 Obsidian vault
用法: python core.py  （启动后常驻监视，Ctrl+C 退出；启动时先处理 inbox 已有文件）
"""
import os
import sys
import json
import time
import hashlib
import re
import gc
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime

if sys.stdout:
    sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内模型镜像
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 镜像站不支持 xet，走普通 HTTP
# 本机 Ollama 请求必须直连：系统环境变量可能残留 HTTP_PROXY(如 127.0.0.1:3145 的代理软件)，
# 绕道代理会让本地长请求(7B 生成长文纪要)被代理掐断超时
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import requests
import json5
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 配置文件定位：环境变量 KETANG_CONFIG 最高优先，其次当前目录/脚本同级/仓库根的 config.json
CONFIG_PATH = None
_env_cfg = os.environ.get("KETANG_CONFIG", "")
for _cand in [
    Path(_env_cfg) if _env_cfg else None,             # 环境变量指定（最高优先）
    Path.cwd() / "config.json",                       # 当前目录（如在项目根运行 python）
    Path(__file__).parent / "config.json",            # 脚本同级（src/config.json）
    Path(__file__).parent.parent / "config.json",     # 仓库根（config.json）
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


def release_whisper():
    """转写模型用完即释放显存。
    根因（2026-09-02 定位）：large-v3 int8 转写后常驻 ~2GB，紧接着 7B 加载 ~4.7GB，
    两模型同驻 ≈7.6GB 逼近 8GB 上限（可用仅 ~6.8GB），llama.cpp KV cache 被挤没 → 生成卡死
    （GPU 100% 十几分钟不返回）。转写后释放，7B 纪要阶段独占显存；下一轮自动重新加载。"""
    global _model
    if _model is not None:
        try:
            del _model
        except Exception:
            pass
        _model = None
        gc.collect()


def fix_terms(text: str, subject: str) -> str:
    """语音识别术语纠正：按学科术语表替换确定性误写（如"柔耻原理"→"容斥原理"）。
    只做高置信替换：词本身在标准语境下几乎必为误写的才入表（config.json term_fixes）。"""
    fixes = CFG.get("term_fixes", {}).get(subject, [])
    if not fixes:
        return text
    for wrong, right in fixes:
        if wrong in text:
            text = text.replace(wrong, right)
    return text


# ---------------- 纪要 + 学科识别 ----------------
def ollama_chat(prompt: str, num_predict: int = None) -> str:
    url = CFG["ollama_host"] + "/api/chat"
    options = {"num_ctx": 49152, "temperature": 0.2}
    if num_predict:
        options["num_predict"] = num_predict  # 输出安全阀：防 7B 话痨无限生成卡死（如单段纪要 4500 token）
    payload = {
        "model": CFG["ollama_model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    }
    r = requests.post(url, json=payload, timeout=1800)
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


def _split_segments(text: str, seg_chars: int = 12000, overlap: int = 1200) -> list:
    """按字符数切段，段间重叠 overlap 字符（避免知识点恰好被切断在边界）。
    12000 字符/段：控制单轮输入(转写片段+模板)在 num_ctx 内，避免超上下文截断丢内容。"""
    lines = text.splitlines()
    segs, cur, cur_len = [], [], 0
    for ln in lines:
        cur.append(ln)
        cur_len += len(ln) + 1
        if cur_len >= seg_chars:
            segs.append("\n".join(cur))
            # 尾部保留 overlap 行作为下段开头，保证衔接
            keep, acc = [], 0
            for l in reversed(cur):
                keep.append(l)
                acc += len(l) + 1
                if acc >= overlap:
                    break
            cur = list(reversed(keep))
            cur_len = acc
    if cur:
        segs.append("\n".join(cur))
    return segs


SUMMARIZE_REQ = """把课堂录音转写整理成详细的知识点罗列笔记。
学科：{subject}｜课程：{course}｜标题：{title}

要求：
- 把课堂内容逐条整理成知识点，宁全勿简：定义、概念、公式、定律、性质、例子、数据、要求、注意事项都尽量保留；
- 篇幅：尽量详尽，转写片段较长时目标 1500-3000 字符，覆盖片段内出现的所有知识点，不要只写梗概；
- 删除口语废话和课堂套话（"同学们好""我们来看""那么""好的""下课"等寒暄连接词），但知识点信息本身要完整；
- 按主题分小节（## 开头），每条知识点用 - 开头；
- 直接从第一个小节开始组织，不要先输出总览/要点列表再重复展开，每个知识点只出现一次；
- 不要写成概括性总结，不要写"本节课主要讲述了…"这类综述，不要写客套结束语。
- 数学公式必须用 LaTeX 写在 $...$（行内）或 $$...$$（独立一行）中，方便 Obsidian 渲染；禁止使用 \\[ \\] 或 \\( \\) 分隔符；
- 术语纠错：本文本来自语音识别，可能存在专业术语谐音误写。同一概念若出现多种写法（如"柔耻原理/柔齿原理/容赤原理"与"容斥原理"并存），统一采用标准学术术语（如"容斥原理"）；明显是术语误写的谐音字（如"单色"实为"单射"、"满色"实为"满射"、"双色"实为"双射"），一律按学科标准术语纠正；

只输出正文，直接写在 SUMMARY_START 之后、SUMMARY_END 之前，不要输出任何说明文字：
SUMMARY_START
SUMMARY_END"""


def _summarize_segment(seg: str, meta: dict, part: str) -> str:
    """整理单个转写片段的知识点草稿（num_predict=4500 防 7B 话痨无限输出卡死）"""
    prompt = f"{SUMMARIZE_REQ.format(**meta)}\n\n这是本节课第 {part} 部分（全课共若干部分），只整理这一部分的内容。\n\n转写片段：\n{seg}"
    raw = ollama_chat(prompt, num_predict=4500)
    m = re.search(r"SUMMARY_START\s*(.*?)\s*SUMMARY_END", raw, re.S)
    return m.group(1).strip() if m else raw.strip()[:8000]


def _draft_sections(text: str):
    """把一段草稿解析成 [(小节标题, [行]), ...]，无标题散行归入 None 标题"""
    sections, cur_title, cur_items = [], None, []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("## "):
            if cur_title is not None or cur_items:
                sections.append((cur_title, cur_items))
            cur_title, cur_items = s[3:].strip(), []
        elif s.startswith("# "):
            # 单井号也当小节标题（部分模型习惯用 #）
            if cur_title is not None or cur_items:
                sections.append((cur_title, cur_items))
            cur_title, cur_items = s[2:].strip(), []
        else:
            cur_items.append(s)  # - 条目/公式行/表格行等一律按原样保留，不丢内容
    if cur_title is not None or cur_items:
        sections.append((cur_title, cur_items))
    return sections


def _text_sim(a: str, b: str) -> float:
    """文本相似度 0-1（SequenceMatcher ratio），用于重复检测"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _dedup_items(items: list) -> list:
    """条目级去重：两条内容相似度 ≥0.85 视为复述，只保留更长的（宁全勿简）"""
    out = []
    for it in items:
        dup = False
        for i, o in enumerate(out):
            if _text_sim(it, o) >= 0.85:
                if len(it) > len(o):
                    out[i] = it
                dup = True
                break
        if not dup:
            out.append(it)
    return out


def _merge_similar_sections(sections: list) -> list:
    """小节级相似归并：标题不同但内容高度相似（相似度 ≥0.45）的小节合并为一个。
    解决 7B 用多个小节标题反复展开同一内容（如"对角线论证方法"连写多节几乎相同）。
    保留先出现的标题；并入小节中与已有内容不重复的条目。"""
    result = []
    for title, lines in sections:
        if not lines:
            continue
        text = "\n".join(lines)
        done = False
        for i, (rt, rlines) in enumerate(result):
            if not done and _text_sim(text, "\n".join(rlines)) >= 0.45:
                keep = [l for l in lines
                        if not any(_text_sim(l, rl) >= 0.85 for rl in rlines)]
                if keep:
                    result[i] = (rt, rlines + keep)
                done = True
        if not done:
            result.append((title, lines))
    return result


def _stitch_drafts(drafts: list) -> str:
    """程序拼接多段独立草稿：同名小节内容归并 + 小节级相似归并 + 条目级去重。
    宁全勿简、内容零丢失（重复由程序消除），返回整门课完整纪要正文。"""
    merged, order, loose = {}, [], []
    for d in drafts:
        for title, lines in _draft_sections(d):
            if title is None:
                loose.extend(lines)
                continue
            if title not in merged:
                merged[title] = []
                order.append(title)
            merged[title].extend(lines)
    sections = [(t, merged[t]) for t in order]
    sections = _merge_similar_sections(sections)
    out_lines, seen_global = [], set()
    for t, lines in sections:
        out_lines.append(f"## {t}")
        for it in _dedup_items(lines):
            # 全局精确去重：完全相同的条目/公式行全文只保留第一次出现（跨小节重复的冗余）
            if it.startswith("- ") or it.startswith("$$"):
                if it in seen_global:
                    continue
                seen_global.add(it)
            out_lines.append(it)
    if loose:
        out_lines.append("## 其他")
        out_lines.extend(_dedup_items(loose))
    return "\n".join(out_lines).strip() + "\n"


def _summarize(full_text: str, meta: dict) -> str:
    """第二步：生成详细知识点罗列正文。
    策略：整门课转写切段 → 每段独立整理成详细草稿（7B 单轮能力内）→ 程序按小节拼接。
    不用模型做长文合并（7B 输出长文极慢且会把转写原文照抄进输出），
    已有内容由程序拼接保证零丢失，宁全勿简、跨段少量重复可接受。"""
    segments = _split_segments(full_text)
    n = len(segments)
    print(f"[纪要] 转写 {len(full_text)} 字符 → 分 {n} 段独立整理 ...", flush=True)
    drafts = []
    for i, seg in enumerate(segments, start=1):
        t0 = time.time()
        draft = _summarize_segment(seg, meta, f"{i}/{n}")
        drafts.append(draft)
        print(f"[纪要] 段 {i}/{n} 草稿 {len(draft)} 字符 (用时 {time.time()-t0:.0f}s)", flush=True)
    summary = _stitch_drafts(drafts)  # 统一拼接：同名/相似小节归并 + 条目去重
    # 公式分隔符强制转 Obsidian 兼容格式（7B 可能不遵守 prompt 的 $$ 要求，这里兜底强制）
    summary = (
        summary.replace(r"\[", "$$")
        .replace(r"\]", "$$")
        .replace(r"\(", "$")
        .replace(r"\)", "$")
    )
    return summary


def generate_note(full_text: str) -> dict:
    print(f"[纪要] 识别学科/课程 ...")
    meta = _classify(full_text)
    # 术语纠正：先按识别出的学科替换确定性误写，再生成纪要（标题也纠正）
    fixed = fix_terms(full_text, meta["subject"])
    title = fix_terms(meta["title"], meta["subject"])
    # 压缩"容斥原理与容斥原理"这类纠正后重复（两个错写都指向同一术语）
    if "与" in title:
        parts = [p for p in title.split("与") if p]
        if len(parts) > 1 and len(set(parts)) == 1:
            title = parts[0]
    meta["title"] = title
    print(f"[纪要] 学科={meta['subject']} 课程={meta['course']} 标题={meta['title']}，生成知识点 ...")
    t0 = time.time()
    summary = _summarize(fixed, meta)
    print(f"[纪要] 完成，用时 {time.time() - t0:.0f}s")
    # 纪要层二次纠错：7B 可能继承转写原文的口音谐音（农事原理/刺客家境/克鲁姆克洛夫等），
    # 输出后按学科术语表整词兜底修正（fix_terms 只替换表内整词，对正确内容无副作用）
    summary = fix_terms(summary, meta["subject"])
    return {**meta, "summary_md": summary, "fixed_text": fixed}


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
    return note_dir / f"{lesson}.md"


# ---------------- 知识补全（归档后自动执行） ----------------
def knowledge_patch(note_path: Path, full_text: str, note: dict):
    """归档后对本课纪要执行知识补全：7B 提取骨架→生成候选→维基查证→幂等写回。
    失败不影响主流程（转写/纪要已归档），异常只打日志。
    维基不可达时 7B 候选不落盘（宁缺毋滥），仅人工定稿（forced_patches.json）可写回。"""
    kcfg = CFG.get("knowledge", {})
    if not kcfg.get("enabled", True):
        print("[补全] knowledge.enabled=false，跳过本课补全")
        return
    if not Path(note_path).exists():
        return
    try:
        import knowledge  # 延迟导入，避免循环依赖
        meta = {
            "subject": (note.get("subject") or "").strip(),
            "course": (note.get("course") or "").strip(),
            "title": (note.get("title") or "").strip(),
        }
        print(f"[补全] 本课知识补全开始（7B 自动）...")
        patches = knowledge.patch_lesson(full_text, meta, Path(note_path).read_text(encoding="utf-8"))
        if patches:
            stats = knowledge.merge_into_note(note_path, patches)
            print(f"[补全] 写回完成: 内嵌 {stats['inserted']} 条，末尾 {stats['appended']} 条，跳过 {stats['skipped']} 条")
        else:
            print("[补全] 本课无可靠补全（宁缺毋滥，不写回）")
    except Exception as e:
        print(f"[补全] 跳过（不影响主流程）: {e}")


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
        release_whisper()  # 释放转写模型显存，7B 独占 GPU（large-v3+7B 同驻超 8GB 会卡死）
        note = generate_note(text)
        fixed = note.pop("fixed_text", text)   # 纠正后的文本给知识补全用（转写文件保留原文）
        note_path = archive(p, text, note, file_hash=h)
        knowledge_patch(note_path, fixed, note)   # 归档后自动知识补全（7B + 维基查证）
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
