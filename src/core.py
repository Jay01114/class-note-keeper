# -*- coding: utf-8 -*-
"""录音整理管家 · 核心流水线
流程: 监视 inbox -> 新音频入队 -> faster-whisper 转写 -> Qwen 会议纪要 -> 写入 Obsidian vault
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

def _app_root() -> Path:
    """定位产品根目录（config/vault/inbox/models 所在层，仅作**兜底**用）。

    注意：正常运行时 inbox/vault/models 的实际路径一律读 config.json 的
    inbox_dir/vault_dir/models_dir，本函数只在 config 缺失或未设该键时兜底。

    层级（2026-09-17 实测）：
    - 源码运行：core.py 在 {root}/app/            → 上溯 1 层 = root
    - 打包运行：exe 在 {root}/课堂笔记管家/        → 上溯 1 层 = root
      （PyInstaller COLLECT 会把 exe 放进同名子目录，所以是 parent 而非 parent.parent）
    - 用户可用环境变量 CNK_ROOT 强制覆盖（便携/多实例）

    2026-09-17：原实现把配置路径硬编码为某个固定盘符下的绝对路径，
    换机器 / 换盘 / 改文件夹名后启动即失败；现改为按上述层级自动定位。
    若既没有 config.json 也没有 config.example.json，会在 import 阶段报错——
    这是刻意为之：配置文件缺失属于部署错误，应尽早暴露。
    """
    env = os.environ.get("CNK_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    if getattr(sys, "frozen", False):
        # exe 所在目录；若该目录没有 config.json（例如直接用 dist/ 下的 exe），
        # 再上溯一层找（兼容 {root}/{app}/exe 与 {root}/exe 两种布局）
        here = Path(sys.executable).resolve().parent
        if (here / "config.json").exists():
            return here
        if (here.parent / "config.json").exists():
            return here.parent
        return here
    return Path(__file__).resolve().parent.parent


CONFIG_PATH = None
_APP_ROOT = _app_root()

# 依次尝试的候选位置（按优先级）。_app_root() 覆盖源码运行与打包运行两种布局；
# 后面两个是兜底：core.py 与 config.json 同目录（打包进 exe 内置模板）、或仓库根目录。
for _base in (_APP_ROOT, Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
    _cand = _base / "config.json"
    if _cand.exists():
        CONFIG_PATH = _cand
        break
if CONFIG_PATH is None:
    CONFIG_PATH = _APP_ROOT / "config.json"

# 首次运行（只放源码、还没配 config.json）时用 config.example.json 兜底，
# 否则 import core 会直接 FileNotFoundError，新用户连启动都看不到提示。
if not CONFIG_PATH.exists():
    for _base in (_APP_ROOT, Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
        _ex = _base / "config.example.json"
        if _ex.exists():
            CONFIG_PATH = _ex
            break


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
    # condition_on_previous_text=False（2026-09-17 根因修复）：
    # 默认 True 时，文件开头若有一段音乐/片尾音频，会把上下文带偏，
    # 后续 78 分钟远场安静人声触发幻觉循环（每 30s 复读一句片尾词）。
    # 实测关闭后可避免长录音后半段重复幻觉（avg_logprob mean=-0.18）。
    segments, info = model.transcribe(path, language="zh", vad_filter=False,
                                      condition_on_previous_text=False)
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


# ---------------- 会议纪要 ----------------
# 瞬时失败重试（2026-09-17 新增）：llama-server 在长 prompt + 多段连续调用下会
# 间歇性返回 HTTP 500（实测同一段重放即 200），旧代码 raise_for_status() 直接抛出，
# 导致「1 段瞬时 500 → 整课 6 段全部作废」。下面按退避重试，且每次重试前探活。
OLLAMA_RETRIES = 4          # 首次 + 3 次重试
OLLAMA_BACKOFF = (2, 5, 10)  # 退避秒数


def _ollama_alive(timeout: float = 3.0) -> bool:
    """探活 /api/tags：区分「服务没起来」和「单次请求偶发 500」。"""
    try:
        r = requests.get(CFG["ollama_host"] + "/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


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
    last_err = None
    for attempt in range(OLLAMA_RETRIES):
        if attempt:
            wait = OLLAMA_BACKOFF[min(attempt - 1, len(OLLAMA_BACKOFF) - 1)]
            print(f"[纪要] 请求失败（{last_err}），{wait}s 后重试 {attempt}/{OLLAMA_RETRIES - 1} ...", flush=True)
            time.sleep(wait)
            if not _ollama_alive():
                # 服务真没了：再等一轮（Ollama 会自己重启 runner），但别把自己拖死
                time.sleep(5)
        try:
            r = requests.post(url, json=payload, timeout=1800)
            if r.status_code >= 500:
                # 5xx = 服务端瞬时故障（runner 重启/上下文槽位回收），可重试
                last_err = f"HTTP {r.status_code}"
                continue
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.exceptions.HTTPError as e:
            # 4xx（如模型不存在）重试无意义，直接抛出
            raise
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                ValueError) as e:
            last_err = type(e).__name__
    raise RuntimeError(f"Ollama 连续 {OLLAMA_RETRIES} 次请求失败（最后错误：{last_err}）")


def _classify(full_text: str) -> dict:
    """提取会议主题和标题，不进行学科或会议类型分类。"""
    excerpt = full_text[:12000]
    prompt = f"""你是会议记录整理员。只根据转写提取一个简短、具体的会议主题和标题。
主题应概括实际讨论内容；信息不足时主题写“未明确”，标题写“未命名会议”。不得猜项目、人名或结论。

只输出两行，不要解释：
TOPIC: 会议主题
TITLE: 简短标题

转写内容：
{excerpt}"""
    raw = ollama_chat(prompt)
    m_topic = re.search(r"TOPIC:\s*(.+)", raw)
    m_t = re.search(r"TITLE:\s*(.+)", raw)
    topic = m_topic.group(1).strip() if m_topic else "未明确"
    title = m_t.group(1).strip() if m_t else "未命名会议"
    return {"topic": topic or "未明确", "title": title or "未命名会议"}


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


SUMMARIZE_REQ = """你是严谨的中文会议记录员。把录音转写整理成忠实、清楚、便于跟进的会议纪要。
会议主题：{topic}｜标题：{title}

只依据本段转写，不联网，不补充常识，不把建议写成已经决定的事项。你的任务是回答“发生了什么事、形成了什么结论、接下来做什么、有什么注意事项”。
口语纠错：只改日常表达中有上下文支持的明显错字、同音误写和重复口头语，不做学科术语纠错；保留原意与不确定语气。人名、客户名、项目名、金额、日期、数字、责任人若听不清，写“待核实”，不得猜测或擅自统一名称。原始转写不改。

按以下固定小节输出；没有信息的小节省略，不能填充“无”或编造内容：
## 发生了什么事
- 按讨论顺序记录关键事实、方案分歧、进展和缘由；区分提议、讨论、已执行。
## 决定与结论
- 只写明确达成的决定；尚未拍板的事写在“待确认问题”。
## 待办事项
- 每项写成“事项：…｜负责人：…｜期限：…”；原文没有负责人或期限时写“未明确”，不能推断。
## 风险与注意事项
- 记录限制条件、依赖、风险、提醒及需要核验的数字或信息；会议日程与下次沟通安排也要保留。
## 待确认问题
- 记录未解决的分歧、待补资料和需要后续确认的问题。

每条只表达一件事，尽量保留重要细节与条件；不要逐字复述寒暄，也不要把不同人的意见合并成一致意见。尽可能标出原文已有的时间戳以便核对；没有可靠时间戳就不添加。禁止臆造发言人身份、金额、截止时间或决议。

只输出正文，直接写在 SUMMARY_START 之后、SUMMARY_END 之前，不要输出任何说明文字：
SUMMARY_START
SUMMARY_END"""


def _summarize_segment(seg: str, meta: dict, part: str) -> str:
    """整理一个会议片段。每段独立提取，避免漏掉后段的新决定和待办。"""
    prompt = (
        f"{SUMMARIZE_REQ.format(**meta)}"
        f"\n\n这是会议录音第 {part} 段。只依据本段记录；后续各段会分别整理并合并。\n\n转写片段：\n{seg}"
    )
    raw = ollama_chat(prompt, num_predict=4500)
    # 2026-09-03 修复：模型输出若无 SUMMARY_END（长输出被 num_predict=4500 截断时常见），
    # 旧代码 raw[:8000] 会把 "SUMMARY_START" 标记原文一起截进来 → 标记泄漏进正文 + 句子腰斩。
    # 改为逐级提取：① 有 END → 取 START..END；② 只有 START → 取 START 之后全部（宁全勿简不截断）；
    # ③ 完全无标记 → 原文直接返回（4500 token 已限长，不会失控）。
    m = re.search(r"SUMMARY_START\s*(.*?)\s*SUMMARY_END", raw, re.S)
    if m:
        body = m.group(1).strip()
    else:
        idx = raw.find("SUMMARY_START")
        body = raw[idx + len("SUMMARY_START"):].strip() if idx >= 0 else raw.strip()

    # 2026-09-18：整段一个 ## 标题都没有 → 判定为不合格草稿。
    # 直接流向下游会被 _draft_sections 收进兜底块，最终纪要先出现一大坨无标题流水
    # （09-17 实测：45 条知识点挤在一起，标题还和内容对不上）。
    # 这里做一次**只补结构、不重写内容**的定向重试：把已有条目原样保留，
    # 只要求模型插小标题；重试仍失败就接受（兜底块接管，不会丢内容）。
    if body and len(body) >= 400 and not re.search(r"^#{1,4}\s+\S", body, re.M):
        try:
            fixed = ollama_chat(
                "下面是一份会议纪要草稿，内容已经写好，但缺少小标题。\n"
                "请只插入以下合适的小节标题：发生了什么事、决定与结论、待办事项、风险与注意事项、待确认问题。\n"
                "硬性要求：\n"
                "- **不得修改、删减、合并、改写任何一条 `- ` 条目**，必须逐条原样保留（含标点）；\n"
                "- 不得新增任何条目，不得补写原文没有的内容；\n"
                "- 只允许插入 `## ` 标题行和调整条目归属；\n"
                "- 不要补充草稿里没有的负责人、期限或决定。\n"
                "- 直接输出整理后的全文，不要任何说明。\n\n"
                f"草稿：\n{body}",
                num_predict=4500,
            )
            m2 = re.search(r"SUMMARY_START\s*(.*?)\s*SUMMARY_END", fixed, re.S)
            f2 = m2.group(1).strip() if m2 else fixed.strip()
            # 只在真的补出了标题、且条目数没被砍（>=90%）时才采纳
            orig_n = body.count("\n- ") + (1 if body.startswith("- ") else 0)
            new_n = f2.count("\n- ") + (1 if f2.startswith("- ") else 0)
            if re.search(r"^#{1,4}\s+\S", f2, re.M) and new_n >= orig_n * 0.9:
                print(f"[整理] 第 {part} 部分原缺小标题，已定向补结构")
                return f2
        except Exception as e:
            print(f"[整理] 第 {part} 部分补结构失败（忽略，用兜底块）: {e}")
    return body


# 无标题散行兜底成多少条一节：7B 整段不给标题时（2026-09-18 实测），
# 若全塞进一个 None 桶会拼成一坨大杂烩；按 8 条切块并配上可追溯标题，
# 至少让结构可读、内容不丢，且用户一眼能看出这段是模型没给标题。
_ORPHAN_CHUNK = 8


def _orphan_sections(lines: list, part: str = "") -> list:
    """把无标题散行切成若干块，每块给一个可追溯的兜底标题。

    标题形如「（未分节内容 · 第2块）」——刻意保留「未分节」字样作为诊断信号，
    方便事后 grep 出哪些课触发了这个降级路径。"""
    lines = [l for l in lines if l.strip()]
    if not lines:
        return []
    out = []
    total = (len(lines) + _ORPHAN_CHUNK - 1) // _ORPHAN_CHUNK
    for i in range(0, len(lines), _ORPHAN_CHUNK):
        chunk = lines[i:i + _ORPHAN_CHUNK]
        n = i // _ORPHAN_CHUNK + 1
        title = "未分节内容"
        if total > 1:
            title += f" · 第{n}块"
        if part:
            title += f"（源自{part}）"
        out.append((title, chunk))
    return out


def _draft_sections(text: str, part: str = ""):
    """把一段草稿解析成 [(小节标题, [行]), ...]，无标题散行归入 None 标题。

    2026-09-17 修复：原实现只认 `## `（恰好两个井号）和 `# `（单井号），
    把 `### ` / `#### ` 当普通内容行吞进条目列表 —— 7B 若整段用三级标题分节
    （实测 09-17 重跑第 1 段如此），该段的全部小节都会塞进一个 None 桶，
    标题以字面文本留在正文里，_stitch_drafts 再把它当散行拼到别处，
    造成「标题与内容错位」。现统一认 `#{1,4}`，与 _summarize 收集 covered
    标题用的正则 `^#{1,4}\\s+` 保持一致。"""
    # 解析：遇到标题就开新节。
    # 2026-09-18：无标题散行不再以 None 形式流向下游。下游 _stitch_drafts 会把 None
    # 当"散行"拼到纪要开头 → 标题与内容错位 + 结构塌缩（09-17 实测：45 条知识点挤在
    # 一起、标题与内容对不上）。现改为切成带兜底标题的小节：结构可读、内容不丢，
    # 且「模型没给标题」显式暴露（标题含"未分节"，便于事后 grep 统计触发率）。
    #
    # 关键点：「标题 → 散行 → 标题」中间的散行，语义上属于**前一个标题**（模型只是
    # 忘了再给一个小标题），不是"整段无标题"。所以这里沿用原有归属逻辑，把散行留在
    # 当前节内；只有"整段从头到尾没有任何标题"时才是真正的无标题草稿，走兜底切块。
    sections, cur_title, cur_items = [], None, []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if "SUMMARY_START" in s or "SUMMARY_END" in s:
            # 防御：模型残留的起止标记行绝不允许进正文（2026-09-03 曾泄漏进"其他"小节）
            continue
        m = re.match(r"^(#{1,4})\s+(.+)$", s)
        if m:
            if cur_title is not None or cur_items:
                sections.append((cur_title, cur_items))
            cur_title, cur_items = m.group(2).strip(), []
        else:
            cur_items.append(s)  # - 条目/公式行/表格行等一律按原样保留，不丢内容
    if cur_title is not None or cur_items:
        sections.append((cur_title, cur_items))

    if any(t for t, _ in sections):
        # 有至少一个标题：散行（None 桶）只可能出现在最前面（正文开头的导语），
        # 用兜底块收容；其余保持"归入当前节"的既有语义不变。
        return [
            (t, items) if t else (f"未分节内容（源自{part}）" if part else "未分节内容", items)
            for t, items in sections
        ]

    # 整段一个标题都没有：切成多块兜底，避免堆成一坨大杂烩
    flat = []
    for _, items in sections:
        flat.extend(items)
    return _orphan_sections(flat, part)


def _dedup_items(items: list) -> list:
    """只删除逐字相同的重复条目，保留措辞相近但条件不同的会议事项。"""
    return list(dict.fromkeys(items))


def _stitch_drafts(drafts: list) -> str:
    """合并分段纪要时按固定小节归并，只去逐字重复，避免吞掉不同条件的会议事项。"""
    merged, order = {}, []
    # 2026-09-18：_draft_sections 现接收 part 标签用于兜底标题可追溯（「未分节内容（源自第2段）」）。
    # 传段号而非变量名 part（此前误传未定义变量导致 NameError）。
    for i, d in enumerate(drafts, 1):
        for title, lines in _draft_sections(d, f"第{i}段"):
            if title not in merged:
                merged[title] = []
                order.append(title)
            merged[title].extend(lines)
    sections = [(t, merged[t]) for t in order]
    out_lines, seen_global = [], set()
    for t, lines in sections:
        out_lines.append(f"## {t}")
        for it in _dedup_items(lines):
            # 完全相同的条目只保留一次；相近事项仍分别保留。
            if it.startswith("- ") or it.startswith("$$"):
                if it in seen_global:
                    continue
                seen_global.add(it)
            out_lines.append(it)
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
    """完整保留会议转写后切段，由 Qwen 分段提取事项，再由程序按小节拼接。"""
    segments = _split_segments(full_text)
    n = len(segments)
    print(f"[纪要] 转写 {len(full_text)} 字符 → 分 {n} 段独立整理 ...", flush=True)
    drafts = []
    failed: list = []    # 失败段号（不阻断整体，最后汇总提示）
    for i, seg in enumerate(segments, start=1):
        t0 = time.time()
        try:
            draft = _summarize_segment(seg, meta, f"{i}/{n}")
        except Exception as e:
            # 单段失败不放弃整课（2026-09-17）：重试已在 ollama_chat 内做过，
            # 到这里说明该段确实反复失败。跳过该段继续后续段，最后如实报告缺口，
            # 避免「段 5 挂了 → 前 4 段成果全丢」。
            failed.append(i)
            print(f"[纪要] 段 {i}/{n} 整理失败（已重试）：{e}", flush=True)
            continue
        drafts.append(draft)
        print(f"[纪要] 段 {i}/{n} 草稿 {len(draft)} 字符 (用时 {time.time()-t0:.0f}s)", flush=True)
    if failed:
        print(f"[纪要] 注意：第 {','.join(map(str, failed))}/{n} 段未能整理，最终纪要缺少这些段落内容", flush=True)
    if not drafts:
        raise RuntimeError(f"全部 {n} 段整理均失败，无法生成纪要")
    summary = _stitch_drafts(drafts)
    # 公式分隔符强制转 Obsidian 兼容格式（7B 可能不遵守 prompt 的 $$ 要求，这里兜底强制）
    summary = (
        summary.replace(r"\[", "$$")
        .replace(r"\]", "$$")
        .replace(r"\(", "$")
        .replace(r"\)", "$")
    )
    return summary


def generate_note(full_text: str) -> dict:
    print("[纪要] 提取会议主题 ...")
    meta = _classify(full_text)
    print(f"[纪要] 主题={meta['topic']} 标题={meta['title']}，整理会议事项 ...")
    t0 = time.time()
    summary = _summarize(full_text, meta)
    print(f"[纪要] 完成，用时 {time.time() - t0:.0f}s")
    return {**meta, "summary_md": summary}


# ---------------- 归档到 Obsidian vault ----------------
def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "未命名"


def unique_lesson(lesson: str, is_taken) -> str:
    """同日同标题冲突时生成唯一 lesson 名（音频/meta/转写/纪要四个文件统一使用）。
    is_taken(名字) 返回 True 表示该名字已被任一目标文件占用。
    冲突时追加可读序号：{lesson}_2、{lesson}_3…（不使用 Unix 时间戳这类不可读后缀）。
    纯函数，便于单测（2026-09-17 修复：此前只对音频做同名兜底，同日同标题的
    转写/纪要/meta 会用原 lesson 名互相覆盖）。"""
    if not is_taken(lesson):
        return lesson
    n = 2
    while is_taken(f"{lesson}_{n}"):
        n += 1
    return f"{lesson}_{n}"


def source_datetime(path: Path) -> tuple:
    """返回归档命名时间和来源；文件名时间仅代表文件名所标时间，不推断会议开始时间。"""
    stem = path.stem
    patterns = (
        (r"(?<!\d)(20\d{2})[-年](\d{1,2})[-月](\d{1,2})日?[ T_-]+(\d{1,2})[:时_-](\d{1,2})(?:[:分_-](\d{1,2}))?",
         "ymd-separated"),
        (r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_ T-](\d{2})(\d{2})(\d{2})?(?!\d)",
         "ymd-compact"),
    )
    for pattern, fmt in patterns:
        m = re.search(pattern, stem)
        if not m:
            continue
        parts = [int(v) if v is not None else 0 for v in m.groups()]
        try:
            return datetime(*parts), f"filename:{fmt}"
        except ValueError:
            continue
    try:
        return datetime.fromtimestamp(path.stat().st_mtime), "file_mtime"
    except OSError:
        return datetime.now(), "processing_time_fallback"


def archive(path: Path, text: str, note: dict, file_hash: str = "") -> Path:
    source_dt, time_source = source_datetime(path)
    date_str = source_dt.strftime("%Y-%m-%d")
    time_str = source_dt.strftime("%H-%M")
    topic = sanitize(note.get("topic") or "未明确")
    title = sanitize(note.get("title") or "未命名会议")
    filename_topic = sanitize(topic if topic != "未明确" else "未命名会议")

    # 结构：会议/{原材料,纪要,转写}/；日期时间来自文件名或文件修改时间。
    vault = Path(CFG["vault_dir"])
    raw_dir = vault / "会议" / "原材料"
    note_dir = vault / "会议" / "纪要"
    trans_dir = vault / "会议" / "转写"
    for d in (raw_dir, note_dir, trans_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 一次性确定唯一最终 lesson 名：移动音频和写任一文档之前先解析，
    # 四个文件（音频/meta.json/转写.md/纪要.md）统一使用，杜绝同日同标题互相覆盖
    def _taken(name: str) -> bool:
        return any([
            (raw_dir / f"{name}{path.suffix}").exists(),
            (raw_dir / f"{name}.meta.json").exists(),
            (note_dir / f"{name}.md").exists(),
            (trans_dir / f"{name}.md").exists(),
        ])

    lesson = unique_lesson(f"{date_str}_{time_str}_{filename_topic}", _taken)

    # 2026-09-17 修复：unique_lesson 通过后到 os.replace 之间仍有竞态窗口（另一课可能
    # 刚创建同名文件），而 os.replace 对已存在目标是**静默覆盖**。这里在写任何文件前
    # 再解析一次「真正可用的最终名」，并把它同时用于音频/meta/转写/纪要四处，
    # 保证 meta["file"] 指向的音频名与实际落盘名一致。
    def _all_free(name: str) -> bool:
        """四处目标名（含 meta）当前都未被占用"""
        return not any([
            (raw_dir / f"{name}{path.suffix}").exists(),
            (raw_dir / f"{name}.meta.json").exists(),
            (note_dir / f"{name}.md").exists(),
            (trans_dir / f"{name}.md").exists(),
        ])

    if not _all_free(lesson):
        n = 2
        while not _all_free(f"{lesson}_{n}"):
            n += 1
        lesson = f"{lesson}_{n}"
        print(f"[归档] 目标名被占用，改用 {lesson}")

    audio_dst = raw_dir / f"{lesson}{path.suffix}"
    transcript_dst = trans_dir / f"{lesson}.md"
    note_dst = note_dir / f"{lesson}.md"
    meta_dst = raw_dir / f"{lesson}.meta.json"
    temp_paths = [
        trans_dir / f".{lesson}.md.part",
        note_dir / f".{lesson}.md.part",
        raw_dir / f".{lesson}.meta.json.part",
    ]
    created_final = []

    transcript = (
        f"# {title}\n\n会议主题：{topic}｜文件时间：{source_dt.isoformat(timespec='minutes')}\n"
        f"> 文件名/修改时间不一定等于会议开始时间。时间来源：{time_source}\n\n{text}\n"
    )
    frontmatter = (
        f"---\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        f"meeting_topic: {json.dumps(topic, ensure_ascii=False)}\n"
        f"file_datetime: {source_dt.isoformat(timespec='minutes')}\n"
        f"time_source: {time_source}\n"
        f"date: {date_str}\n"
        f"tags: [会议纪要]\n"
        f"---\n\n"
    )
    summary_md = note.get("summary_md") or ""
    if not summary_md:
        summary_md = "> 未生成纪要；转写原文已归档。\n"
    summary_md = (
        f"# {title}\n\n会议主题：{topic}｜文件时间：{source_dt.isoformat(timespec='minutes')}\n"
        f"> 文件名/修改时间不一定等于会议开始时间。时间来源：{time_source}\n\n"
        f"{summary_md}"
    )
    meta = {
        "file": audio_dst.name,
        "topic": topic,
        "title": title,
        "date": date_str,
        "file_datetime": source_dt.isoformat(timespec="minutes"),
        "time_source": time_source,
        "file_time_is_meeting_start": False,
        "file_hash": file_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    # 先在目标目录同卷写入临时文件，再移动音频，最后用 os.replace 原子切换。
    # 任一步失败都清理临时文件并尽力把音频移回原 inbox，避免留下半归档。
    moved_audio = False
    try:
        # 2026-09-17 修复：先移音频再写临时文件。原顺序下「临时文件写入失败」
        # 会把已移入 vault 的音频回滚（备份目录还多一份），且 temp_paths[2] 失败时
        # 回滚循环里 final 可能被 unlink —— 先移音频后写文件彻底避开。
        shutil.move(str(path), str(audio_dst))
        moved_audio = True
        temp_paths[0].write_text(transcript, encoding="utf-8")
        temp_paths[1].write_text(frontmatter + summary_md, encoding="utf-8")
        temp_paths[2].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        for tmp, final in zip(temp_paths, (transcript_dst, note_dst, meta_dst)):
            if final.exists():
                # 最后一道防线：_all_free 之后到这里的极窄窗口内被抢占，
                # 宁可报错让本课进 _失败 目录，也绝不覆盖别人的笔记
                raise FileExistsError(f"归档目标已被占用（拒绝覆盖）：{final}")
            os.replace(str(tmp), str(final))
            created_final.append(final)
    except Exception:
        for tmp in temp_paths:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        for final in created_final:
            try:
                if final.exists():
                    final.unlink()
            except Exception:
                pass
        if moved_audio and audio_dst.exists() and not path.exists():
            try:
                shutil.move(str(audio_dst), str(path))
            except Exception as rollback_error:
                print(f"[归档] 回滚音频失败：{rollback_error}", flush=True)
        raise

    print(f"[归档] -> {vault / '会议'}（原材料/纪要/转写）")
    return note_dst


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
        try:
            text = transcribe(str(p))
        finally:
            # 转写成功或异常都释放模型，避免下一次纪要/转写继续占用显存。
            release_whisper()
        note = generate_note(text)
        note_path = archive(p, text, note, file_hash=h)
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


def seen_cleanup(seen: set, stable: dict, name: str):
    """处理完一个文件后清理其黑名单与稳定记录（2026-09-17 修复）。
    此前 seen 以文件名为键且只增不减：处理完 A.m4a 后，若 inbox 再次出现
    内容不同但同名的 A.m4a，会被永久静默跳过。清理后，同名同大小的新文件
    仍必须重新经过完整 STABLE_SECONDS 稳定检测（stable 也要清，防止新文件
    直接沿用旧 (size, first_seen) 而跳过稳定等待）。"""
    seen.discard(name)
    stable.pop(name, None)


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
                if STOP_FLAG:
                    break
                if f.name in seen:
                    continue
                if is_ready_audio(f, stable, now):
                    seen.add(f.name)
                    process_one(str(f))
                    seen_cleanup(seen, stable, f.name)
            time.sleep(3)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[监视异常] {e}")
            time.sleep(5)


if __name__ == "__main__":
    main_loop()
