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
    # 2026-09-03：长词优先（Python sorted 稳定，同长保持配置顺序），
    # 防止短规则先命中、把长规则待匹配串改掉
    fixes = sorted(fixes, key=lambda p: len(p[0]), reverse=True)
    for wrong, right in fixes:
        if wrong in text:
            text = text.replace(wrong, right)
    return text


# ---------------- 标准术语词库（term_dict/*.json → prompt 锚点注入） ----------------
# 2026-09-03 接入：四科词库由本地 qwen2.5:7b 按章节生成 + 维基校验打标（term_glossary.py 产物），
# 存 {term_dict_dir}/{学科}.json（terms=[{ch,t,v}]）。用途：把标准术语名单注入 _summarize /
# knowledge 补全 prompt，让 7B 遇到音近词时按标准写法书写（软约束；fix_terms 仍是硬兜底）。
# 词库文件缺失时静默降级为空（不注入），不影响主流程。


def _term_dict_path(subject: str) -> Path:
    """定位 {subject}.json：config 显式 term_dict_dir 优先，其次 app 同级/项目根。"""
    rel = (CFG.get("term_dict_dir") or "").strip()
    cands = []
    if rel:
        p = Path(rel)
        cands.append(p if p.is_absolute() else CONFIG_PATH.parent / p)
    cands += [
        Path(__file__).parent / "term_dict",
        Path(__file__).parent.parent / "term_dict",
    ]
    for c in cands:
        f = c / f"{subject}.json"
        if f.exists():
            return f
    return cands[-1] / f"{subject}.json"


def _load_subject_terms(subject: str) -> list:
    """读词库 → [(term, 校验层 v, 章节 ch)]；文件不存在/损坏返回空（不抛异常）。"""
    try:
        p = _term_dict_path(subject)
        if not p.exists():
            return []
        d = json.load(open(p, encoding="utf-8"))
        terms = d.get("terms") or []
        return [(t["t"], t.get("v", ""), t.get("ch", "")) for t in terms
                if isinstance(t, dict) and t.get("t")]
    except Exception:
        return []


_term_guide_cache = {}

# 2026-09-03：每科「核心保底词」——课堂高频且音近易错的标准术语，硬性进入锚点名单。
# 键为词库中词条的关键词（支持精确/包含匹配，词库缺失则跳过不报错）。
_GUIDE_MUST = {
    "数字逻辑和计算机组成": ["真值表", "卡诺图", "与非门", "或非门", "触发器", "Verilog",
                    "寄存器", "补码", "最小项", "最大项", "译码器", "全加器", "锁存器", "总线"],
    "概率论": ["随机变量", "概率密度", "分布函数", "正态分布", "大数定律", "中心极限定理",
               "数学期望", "方差", "协方差", "贝叶斯", "假设检验", "置信区间", "泊松", "二项分布"],
    "离散数学": ["单射", "满射", "双射", "容斥原理", "等价关系", "偏序", "数学归纳法",
               "欧拉图", "哈密顿", "命题", "谓词", "幂集", "笛卡尔积"],
    "电路、信号和系统": ["基尔霍夫", "欧姆定律", "叠加定理", "戴维南", "诺顿", "傅里叶变换",
                 "拉普拉斯变换", "Z变换", "卷积", "采样定理", "滤波器", "传递函数", "相量", "阻抗"],
}


def _term_guide(subject: str) -> str:
    """按学科生成术语锚点段（2~6 字词，wiki 层优先，≤term_inject_max 条，按章节均衡）。
    2026-09-03 改：①原按词库顺序取前 N 个 wiki 词会被大章节占满名额 → 章节轮转采样；
    ②再加 _GUIDE_MUST 核心词保底置顶（含 6~7 字长词如"拉普拉斯变换/Verilog"，避开长度过滤）。
    无词库/词条不足时返回 ""（调用方拼接时跳过）。"""
    if not subject:
        return ""
    if subject in _term_guide_cache:
        return _term_guide_cache[subject]
    max_n = int(CFG.get("term_inject_max", 80) or 80)
    pairs = _load_subject_terms(subject)
    # 1) 核心保底词（按关键词精确→包含最短匹配，去重置顶）
    must_keys = _GUIDE_MUST.get(subject, [])
    picked = []
    for key in must_keys:
        hit = None
        for t, v, ch in pairs:
            if t == key:
                hit = t
                break
        if hit is None:
            best = [t for t, v, ch in pairs if key in t]
            if best:
                hit = min(best, key=len)
        if hit and hit not in picked:
            picked.append(hit)
    # 2) 其余名额：按章节分组（章内 wiki 优先、稳定排序），轮转采样补足
    groups = {}
    order = []
    for t, v, ch in pairs:
        if not (2 <= len(t) <= 6):
            continue
        if t in picked:
            continue
        if ch not in groups:
            groups[ch] = []
            order.append(ch)
        groups[ch].append((t, v))
    for ch in order:
        groups[ch].sort(key=lambda x: 0 if x[1] == "wiki" else 1)
    idxs = {ch: 0 for ch in order}
    while len(picked) < max_n:
        progressed = False
        for ch in order:
            arr = groups[ch]
            j = idxs[ch]
            if j < len(arr):
                picked.append(arr[j][0])
                idxs[ch] = j + 1
                progressed = True
                if len(picked) >= max_n:
                    break
        if not progressed:
            break
    txt = ""
    if picked:
        txt = (
            "本课可能涉及以下标准术语。语音转写会把术语写成谐音错字，"
            "凡读音相近、写法拿不准处，一律按本名单的标准写法书写：\n"
            + "、".join(picked)
        )
    _term_guide_cache[subject] = txt
    return txt


# ---------------- 纪要 + 学科识别 ----------------
def ollama_chat(prompt: str, num_predict: int = None) -> str:
    url = CFG["ollama_host"] + "/api/chat"
    # num_ctx=16384（2026-09-03 调低）：7B Q4 显存账 4.7GB + 16k ctx KV ~1.3GB + CUDA 开销
    # ≈ 6.4GB，8GB 卡留 ~1.6GB 安全冗余。此前 49152 的 KV ~3.9GB → 合计 ~9GB 超显存，
    # 长录音分多段连续调用时 GPU 逐步耗尽，段 3 后降速、段 4 直接 1800s read timeout 卡死。
    options = {"num_ctx": 16384, "temperature": 0.2}
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


# ---- 低信息区清理（2026-09-07 新增）----
# 背景：老师指着板书/屏幕讲例题时，语音只有碎片指示词与无意义音译（如
# "关了20V…照着这个IE…rete off fait IЬ insist"），喂给 7B 只会诱导它编造
# 变量名/方程/数值（伪精确比留白更毒，见电路 09-07 纪要 $IE-IO-I3-I4-I5=0$）。
# 策略：入纪要 prompt 前先清掉纯口语填充行、把乱码/填充密集区折叠成占位行。
# 保守设计：只处理带 [hh:mm:ss] 前缀的转写行；乱码需连续 ≥3 行、填充需成片才折叠。
_FILLER_TOKENS = sorted([
    "这个", "那个", "然后", "就是", "可以", "我们", "你们", "他们", "大家", "你看",
    "来看", "看一下", "对不对", "是不是", "怎么样", "是吧", "好吧", "行了", "好的", "对吧",
    "嗯嗯", "啊啊", "哦哦", "嗯", "哦", "啊", "呃", "哈", "呐", "哎", "嘛", "对", "好",
    "是", "吧", "吗", "行", "那", "这", "它", "你", "我", "他", "她", "了",
], key=len, reverse=True)

_TS_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")


def _ts_sec(hh: str, mm: str, ss: str) -> int:
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def _is_filler_line(body: str) -> bool:
    """整行都是口语填充/语气词（≤8 字）→ True。超过 8 字或含任何实词一律不算（防误伤）。"""
    b = re.sub(r"[\s，。！？、；：,.!?;:'\"()\[\]·…\-—]", "", body)
    if not b or len(b) > 8:
        return False
    rest = b
    while rest:
        hit = next((w for w in _FILLER_TOKENS if rest.startswith(w)), None)
        if not hit:
            return False
        rest = rest[len(hit):]
    return True


def _is_garbled_line(body: str) -> bool:
    """乱码/无意义音译行：汉字占比过低或夹杂大量字母符号（"rete off fait 工作室 insist"）。
    排除纯 ASCII 单 token 术语行（Verilog/KCL/PDF）与正常短句（汉字 ≥40%）。"""
    b = body.strip()
    if not b:
        return False
    han = sum(1 for ch in b if "\u4e00" <= ch <= "\u9fff")
    if han * 10 >= len(b) * 4:      # 汉字占比 ≥40% → 视为正常行
        return False
    if re.fullmatch(r"[A-Za-z0-9_.\-/+]{1,24}", b):   # 纯 ASCII 术语/编号
        return False
    letters = sum(1 for ch in b if ch.isascii() and ch.isalpha())
    return letters >= 2 or han == 0


def _collapse_lowinfo(text: str) -> str:
    """转写文本低信息区清理，返回清理后的文本（供 generate_note 的纪要环节使用，
    不影响归档的转写原文与知识补全的锚点原文）。"""
    lines = text.splitlines()
    out, n = [], len(lines)
    i = 0
    while i < n:
        ln = lines[i]
        m = _TS_RE.match(ln)
        if not m:
            out.append(ln)
            i += 1
            continue
        body = m.group(4).strip()
        # 1) 乱码密集区（≥3 行连续乱码）→ 折叠占位
        if _is_garbled_line(body):
            j, start = i, _ts_sec(*m.group(1, 2, 3))
            while j < n:
                mm = _TS_RE.match(lines[j])
                if not (mm and _is_garbled_line(mm.group(4).strip())):
                    break
                j += 1
            if j - i >= 3:
                end = _ts_sec(*_TS_RE.match(lines[j - 1]).group(1, 2, 3))
                dur = max(round((end - start) / 60), 1)
                out.append(f"[{m.group(1)}:{m.group(2)}:{m.group(3)}]"
                           f"（此段约 {dur} 分钟为板书/画面讲解或语音不清，题目与过程无法从语音还原，省略不整理）")
                i = j
                continue
            # 不足 3 行的散乱码：原样保留（可能是短术语/编号，宁全勿删）
            out.append(ln)
            i += 1
            continue
        # 2) 纯填充行：成片（≥6 行）折叠占位，零星则直接删除（无信息）
        if _is_filler_line(body):
            j = i
            while j < n:
                mm = _TS_RE.match(lines[j])
                if not (mm and _is_filler_line(mm.group(4).strip())):
                    break
                j += 1
            if j - i >= 6:
                end = _ts_sec(*_TS_RE.match(lines[j - 1]).group(1, 2, 3))
                start = _ts_sec(*m.group(1, 2, 3))
                dur = max(round((end - start) / 60), 1)
                out.append(f"[{m.group(1)}:{m.group(2)}:{m.group(3)}]"
                           f"（此段约 {dur} 分钟为口头过渡/寒暄，无实质内容，省略）")
            i = j
            continue
        out.append(ln)
        i += 1
    removed = len(lines) - len(out)
    if removed:
        print(f"[纪要] 低信息区清理：删除/折叠 {removed} 行（{len(lines)} → {len(out)} 行）", flush=True)
    return "\n".join(out)


def _split_segments(text: str, seg_chars: int = 8000, overlap: int = 1200) -> list:
    """按字符数切段，段间重叠 overlap 字符（避免知识点恰好被切断在边界）。
    8000 字符/段（2026-09-03 调低）：配合 num_ctx=16384 —— 单段 ~8000 字符转写
    (≈4500-6000 tokens) + 指令模板 ~1000 + 输出上限 4500 ≤ ~11500 < 16384，不截断。
    此前 12000 字符/段 + num_ctx=49152 显存超载导致长录音第 4 段 1800s 卡死。"""
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
- 课堂问答/互动环节的口语原句（师生对话、提问应答、"这个考不考""要不要记"这类来回）不要原句照搬成条目，要把其中的知识信息提炼成规范表述（如"考试重点：XX"）；去掉问答外壳，保留知识点本身；
- 数学公式必须用 LaTeX 写在 $...$（行内）或 $$...$$（独立一行）中，方便 Obsidian 渲染；禁止使用 \\[ \\] 或 \\( \\) 分隔符；
- 术语纠错：本文本来自语音识别，可能存在专业术语谐音误写。同一概念若出现多种写法（如"柔耻原理/柔齿原理/容赤原理"与"容斥原理"并存），统一采用标准学术术语（如"容斥原理"）；明显是术语误写的谐音字（如"单色"实为"单射"、"满色"实为"满射"、"双色"实为"双射"），一律按学科标准术语纠正；
- 口语过程不转述：老师的口头推理过程、自问自答、寒暄过渡（如"为什么要…呢""我们来看一下""实际上是这样子的""它就会怎么怎么样"）不要逐句转写成条目；只能提炼成有完整结论的知识点。禁止输出"通过某种方式来实现""与…类似""跟…有关系"这类没有结论的空句——每条知识点脱离上下文必须能单独读懂、给出明确结论（定义/公式/性质/结论本身）；
- 复习内容压缩：若片段开头或中间包含对本课程上一节的复习（如"我们先把上节课的内容复习一下"），复习部分不要按时间线逐句罗列，只提炼成 1~2 条背景知识点（如"复习：上节的 XX 概念/方法"）作为铺垫，把篇幅留给本节课的新内容；新旧知识靠小标题区分，不要混排。
- 例题/讲题处理（重要）：课堂讲例题时，若语音讲全了题目（已知条件+求解目标都有）→ 整理成完整条目「例：题干…｜方法…｜结论/答案…」，题设里的数字/符号只能来自转写原文，语音没讲到的不得补写；若一段讲题语音只有碎片指示词（"这个""关了 20V""照着这个列"）或明显是老师指着板书/屏幕边讲边比划、语音里没有完整题设 → 该例题整体省略，最多留一句"课上用板书讲解了一道 XX 型例题（题干在板书，未录音）"，禁止编造变量名/方程/数值/答案——凡语音支撑不了的"列出方程…=0""解得…"一律丢弃；
- 宁缺毋滥：只保留能脱离上下文独立读懂、有明确结论的条目。删掉"老师介绍了/说明了/强调了 XX"却没写出 XX 内容的空句、"本节总结了/讨论了/回顾了…"这类元描述、与前面条目重复的表述；同一主题的内容只允许出现在一个小节，老师换例子重讲/补充时并入已有小节，不得另起小节换标题反复展开；小节标题不要写成课程总标题或与文件名相同的名字。

只输出正文，直接写在 SUMMARY_START 之后、SUMMARY_END 之前，不要输出任何说明文字：
SUMMARY_START
SUMMARY_END"""


def _summarize_segment(seg: str, meta: dict, part: str, covered: tuple = ()) -> str:
    """整理单个转写片段的知识点草稿（num_predict=4500 防 7B 话痨无限输出卡死）。
    covered：此前各段已整理出的小节标题（去重防段间重复：段 2 不知道段 1 写了什么，
    常把同一概念换措辞再开一节重写，导致最终纪要同主题重复展开）。"""
    guide = _term_guide(meta.get("subject", ""))
    guide_blk = f"\n\n{guide}" if guide else ""
    covered_blk = ""
    if covered:
        covered_blk = (
            "\n\n前面部分已经整理过这些主题（小节标题）：\n"
            + "、".join(dict.fromkeys(covered))
            + "\n请勿再为这些主题另起小节重复展开定义/定理；"
            "若本部分出现同一主题，只写前面没写过的补充内容（新例子、新结论、细节），"
            "或一句话说明它是复习/强调即可。"
        )
    prompt = (
        f"{SUMMARIZE_REQ.format(**meta)}{guide_blk}{covered_blk}"
        f"\n\n这是本节课第 {part} 部分（全课共若干部分），只整理这一部分的内容。\n\n转写片段：\n{seg}"
    )
    raw = ollama_chat(prompt, num_predict=4500)
    # 2026-09-03 修复：模型输出若无 SUMMARY_END（长输出被 num_predict=4500 截断时常见），
    # 旧代码 raw[:8000] 会把 "SUMMARY_START" 标记原文一起截进来 → 标记泄漏进正文 + 句子腰斩。
    # 改为逐级提取：① 有 END → 取 START..END；② 只有 START → 取 START 之后全部（宁全勿简不截断）；
    # ③ 完全无标记 → 原文直接返回（4500 token 已限长，不会失控）。
    m = re.search(r"SUMMARY_START\s*(.*?)\s*SUMMARY_END", raw, re.S)
    if m:
        return m.group(1).strip()
    idx = raw.find("SUMMARY_START")
    if idx >= 0:
        return raw[idx + len("SUMMARY_START"):].strip()
    return raw.strip()


def _draft_sections(text: str):
    """把一段草稿解析成 [(小节标题, [行]), ...]，无标题散行归入 None 标题"""
    sections, cur_title, cur_items = [], None, []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if "SUMMARY_START" in s or "SUMMARY_END" in s:
            # 防御：模型残留的起止标记行绝不允许进正文（2026-09-03 曾泄漏进"其他"小节）
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
    # 2026-09-07：吸收「总结/小结/回顾」类冗余节——7B 常违反"不写综述"在末尾堆一节
    # 复述前文。处理：与前文条目相似 ≥0.35 的重复条目直接丢弃；确实新增的低相似条目
    # （如"考试重点"）并入前一小节末尾，不单独立节。
    _ABSORB_TITLES = ("总结", "小结", "回顾", "综述", "收尾")
    absorbed = []
    for t, lines in sections:
        if any(k in t for k in _ABSORB_TITLES):
            prev_all = [l for _, ls in absorbed for l in ls]
            fresh = [l for l in lines
                     if not any(_text_sim(l, p) >= 0.35 for p in prev_all)]
            if fresh:
                if absorbed:
                    absorbed[-1] = (absorbed[-1][0], absorbed[-1][1] + fresh)
                else:
                    absorbed.append((t, fresh))
            continue
        absorbed.append((t, lines))
    sections = absorbed
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
    # 2026-09-07 兜底：任何路径残留的起止标记行一律剔除（此前曾泄漏进成品末尾）
    out_lines = [l for l in out_lines if "SUMMARY_START" not in l and "SUMMARY_END" not in l]
    # 剔除孤立标题行：小节标题后没有任何条目（7B 输出被截断留下的空节标题）
    cleaned, pend, pend_has_item = [], None, False
    for ln in out_lines:
        if re.match(r"^#{2,4}\s", ln):
            if pend is not None and pend_has_item:
                cleaned.append(pend)
            pend, pend_has_item = ln, False
        else:
            if pend is not None:
                cleaned.append(pend)
                pend = None
            pend_has_item = True   # 有非标题行 → 说明此前/当前标题有内容
            cleaned.append(ln)
    if pend is not None and pend_has_item:
        cleaned.append(pend)
    out_lines = cleaned
    return "\n".join(out_lines).strip() + "\n"


def _summarize(full_text: str, meta: dict) -> str:
    """第二步：生成详细知识点罗列正文。
    策略：整门课转写切段 → 每段独立整理成详细草稿（7B 单轮能力内）→ 程序按小节拼接。
    不用模型做长文合并（7B 输出长文极慢且会把转写原文照抄进输出），
    已有内容由程序拼接保证零丢失，宁全勿简、跨段少量重复可接受。"""
    full_text = _collapse_lowinfo(full_text)   # 2026-09-07：先清板书碎片/乱码/口语填充，防 7B 编造
    segments = _split_segments(full_text)
    n = len(segments)
    print(f"[纪要] 转写 {len(full_text)} 字符 → 分 {n} 段独立整理 ...", flush=True)
    drafts = []
    covered: list = []   # 已整理小节标题（喂后续段防同主题重复展开）
    for i, seg in enumerate(segments, start=1):
        t0 = time.time()
        draft = _summarize_segment(seg, meta, f"{i}/{n}", tuple(covered))
        drafts.append(draft)
        # 增量收集本段草稿的小节标题（任意 ##/### 层级，供后续段去重参考）
        for raw_t in re.findall(r"^#{1,4}\s+(.+?)\s*$", draft, re.M):
            t = re.sub(r"^\d+[.、)）\s]+", "", raw_t).strip()   # 去"1."序号，同主题更好匹配
            if t and len(t) <= 24 and t not in covered:
                covered.append(t)
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
