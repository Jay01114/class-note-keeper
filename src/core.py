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
    # 实测关闭后同一文件转出 1842 段正常课程内容（avg_logprob mean=-0.18）。
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
    course = m_c.group(1).strip() if m_c else subject
    title = m_t.group(1).strip() if m_t else "未命名课程"

    # 关键词投票纠偏（见 _subject_vote 注释）
    voted, top_n, second_n = _subject_vote(full_text)
    if voted and voted != subject and top_n >= _SUBJECT_VOTE_MIN and \
            top_n >= _SUBJECT_VOTE_RATIO * max(second_n, 1):
        print(f"[纪要] 学科关键词投票纠偏：{subject} → {voted}（{top_n} 票 vs 次高 {second_n} 票）")
        subject = voted
        # 模型顺着错误学科编的课程名必须一起丢弃（如「数字逻辑与密码学基础」）
        course = voted

    return {"subject": subject, "course": course, "title": title}


# ---- 学科关键词投票（2026-09-16 新增）----
# 背景：7B 分类对某些课程有稳定偏见，不是随机抽风。实测 09-16「费马小定理与欧拉定理」
# （纯数论，应属离散数学）连跑 3 次都判成「数字逻辑和计算机组成」，且改成
# 「开头+中段+尾段」三段采样仍然判错 —— 说明是模型偏见，不是信息量不足，喂更多文本没用。
# 对策：用一份「学科专属特征词」表对全文投票，票数压倒性时直接覆盖模型结论。
# 只收各科高专有度词（「费马小定理」「卡诺图」），不收「逻辑」「证明」这类跨科词，
# 避免误伤。实测：vault 内 20 门历史课程全部判对（零误伤），09-16 给出 88:0。
_SUBJECT_VOTE_MIN = 5      # 最高票下限：太低说明该课不在词表覆盖范围，交回模型判断
_SUBJECT_VOTE_RATIO = 3    # 最高票须≥次高票 3 倍，否则视为有歧义、不覆盖

_SUBJECT_KEYWORDS = {
    "离散数学": [
        "费马小定理", "欧拉定理", "欧拉函数", "同余", "模运算", "素数", "质数", "互质",
        "整除", "约数", "最大公因数", "最小公倍数", "容斥原理", "数学归纳法", "鸽巢原理",
        "拉姆齐", "布尔格", "偏序", "等价关系", "斯特林数", "生成函数", "递推关系",
        "图论", "哈密顿", "欧拉回路", "二项式", "排列组合", "抽屉原理", "唯一分解",
        "强归纳法", "反链", "极大链", "素因子", "合数", "整除性",
    ],
    "数字逻辑和计算机组成": [
        "卡诺图", "真值表", "逻辑门", "与门", "或门", "非门", "异或门", "同或门",
        "触发器", "寄存器", "多路选择器", "译码器", "编码器", "加法器", "计数器",
        "时序逻辑", "组合逻辑", "最小项", "最大项", "布尔代数", "原码", "反码",
        "补码", "数制", "十六进制", "Verilog", "冯诺依曼", "指令周期", "格雷码",
        "奇偶校验", "选择器", "锁存器", "全加器", "半加器", "状态机",
    ],
    "概率论": [
        "随机变量", "概率密度", "分布函数", "期望", "方差", "协方差", "条件概率", "贝叶斯",
        "全概率", "独立同分布", "正态分布", "泊松", "几何分布", "二项分布", "均匀分布",
        "大数定律", "中心极限定理", "假设检验", "样本空间", "互斥", "古典概型", "几何概型",
        "分布律", "无记忆性", "相关系数", "边缘分布", "联合分布", "置信区间",
    ],
    "电路、信号和系统": [
        "基尔霍夫", "欧姆定律", "节点分析", "网孔", "戴维南", "诺顿", "叠加定理",
        "电容", "电感", "阻抗", "谐振", "滤波器", "傅里叶", "拉普拉斯", "Z变换",
        "卷积", "采样定理", "相量", "功率因数", "最大功率传输", "受控源", "运算放大器",
        "受控电源", "电流源", "电压源", "回路电流", "电路分析", "信号处理", "频谱",
    ],
}


def _subject_vote(text: str) -> tuple:
    """按学科专属特征词对全文投票，返回 (学科, 最高票, 次高票)。全零返回 (None, 0, 0)。

    用全文而非开头截断段：偏见的成因之一正是只看开头，全文投票信号强得多。
    """
    try:
        scores = {s: sum(text.count(k) for k in ks) for s, ks in _SUBJECT_KEYWORDS.items()}
    except Exception:
        return None, 0, 0
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    if not ranked or ranked[0][1] <= 0:
        return None, 0, 0
    top, top_n = ranked[0]
    second_n = ranked[1][1] if len(ranked) > 1 else 0
    return top, top_n, second_n


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
- **必须分小节（硬性要求）**：每个小节以 `## ` 开头单独一行，小节内每条知识点用 `- ` 开头。
  正文里**不允许出现不属于任何小节的 `- ` 条目**——第一条 `- ` 之前必须先有一个 `## ` 标题行。
  正确示例：
  ## 主从D触发器的结构
  - 由两个D锁存器串联构成，主级和从级由同一个 CLK 控制。
  - 主级在 CLK=0 时跟随输入 D。
  ## 建立时间与保持时间
  - 建立时间：CLK 边沿到来前，D 必须保持稳定的最短时间。
  错误示例（不允许）：
  - 两个D锁存器串联构成主从结构。
  - 主级在 CLK=0 时跟随输入 D。
  （上面这种「只有条目、没有 ## 标题」的输出视为不合格）
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
                "下面是一份课堂笔记草稿，内容已经写好，但**缺少小标题**。\n"
                "请只做一件事：把它按主题切成若干小节，给每个小节加一行 `## 小标题`。\n"
                "硬性要求：\n"
                "- **不得修改、删减、合并、改写任何一条 `- ` 条目**，必须逐条原样保留（含标点）；\n"
                "- 不得新增任何条目，不得补写原文没有的内容；\n"
                "- 只允许插入 `## ` 标题行和调整条目归属；\n"
                "- 标题要具体（写清是什么电路/什么概念），不要写课程名或「知识回顾」这类空标题。\n"
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


# 2026-09-17：0.45 阈值实测把 09-17 课压成 4 个小节（历史均值 12.8）。
# 7B 对同一阶段会输出「D触发器的工作原理」「D触发器的工作过程」这类泛化小标题，
# 彼此内容相似度天然落在 0.5~0.7，0.45 一网打尽 → 结构塌缩。
# 提到 0.60：仍能吃掉「同标题近重复」（>0.7）与真正复述（>0.85），
# 泛化小标题因内容各有侧重而被保留。复述兜底交给 _dedup_items（0.85 条目级）。
_SECTION_MERGE_SIM = 0.60

# 2026-09-18：0.60 单阈值放过了「同义标题」小节——7B 常把同一主题写成
# 「边沿敏感触发器」「边沿敏感触发器与电平敏感触发器的区别」
# 「电平敏感触发器与边沿敏感触发器的区别」三节（内容各有侧重 → 相似度仅 0.5 上下）。
# 这里的判定不看内容、只看标题：剥掉虚词后词集高度重合即为同义主题，
# 此时阈值放宽到 0.35（内容再怎么侧重，也确实是同一块知识）。
_TITLE_STOP = ("与", "和", "的", "及", "以及", "区别", "对比", "比较",
               "原理", "概念", "介绍", "功能", "实现", "设计", "方法",
               "关系", "作用", "应用", "分析")


def _title_core(title: str) -> str:
    """剥掉标点与虚词，得到用于同义判定的标题主干（纯字符串处理，无副作用）"""
    t = re.sub(r"[，,、（）()\[\]【】:：;；!！?？\"'“”‘’\s]+", "", title or "")
    for w in _TITLE_STOP:
        t = t.replace(w, "")
    return t


def _title_similar(a: str, b: str) -> bool:
    """判断两个小节标题是否指向同一主题。

    判定用「主干相同或互为子串」+ 「词集相同（语序无关）」，
    不接受纯相似度兜底 —— 相似度会把「泛化标题A」和「泛化标题B」
    这类只差末尾一个字母的不同主题判成同义（2026-09-18 回归测试实测）。

    另外要求：短主干（≤4 字）只允许完全相等，防止「寄存器」吃掉一切。"""
    ca, cb = _title_core(a), _title_core(b)
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    # 主干很短时不做包含判断（「寄存器」是「环形计数器」的子串但不该合并）
    if len(ca) <= 4 or len(cb) <= 4:
        return False
    if ca in cb or cb in ca:
        return True
    # 语序无关的字频对比：词集完全相同 → 同义（A与B / B与A）
    return sorted(ca) == sorted(cb)


def _merge_similar_sections(sections: list) -> list:
    """小节级相似归并：标题不同但内容高度相似（相似度 ≥0.60）的小节合并为一个。
    解决 7B 用多个小节标题反复展开同一内容（如"康诺对角线论证方法"连写 4 节几乎相同）。
    保留先出现的标题；并入小节中与已有内容不重复的条目。
    2026-09-17 防塌缩：①阈值 0.45→0.60；②两边条目都不足 2 条时不合并
    （防止 1~2 条「补：…」小节的相似度恰好落在阈值上被误并）；③同名小节不合并。"""
    result = []
    for title, lines in sections:
        if not lines:
            continue
        text = "\n".join(lines)
        done = False
        for i, (rt, rlines) in enumerate(result):
            if rt == title:            # 同名由 _stitch_drafts 的 merged 阶段合并，这里不碰
                continue
            if len(lines) < 2 or len(rlines) < 2:
                continue
            # 2026-09-18：同义标题直接归并，不再看内容相似度。
            # 依据：标题（剥虚词后）指向同一主题，就是同一块知识——7B 常把
            # 「边沿敏感触发器」拆成「…与电平敏感触发器的区别」正反两节，
            # 两节内容天然互补（各写一半），相似度只有 0.2 上下，
            # 若还按内容阈值判就永远合不掉 → 主题碎裂。
            # 反向风险（误并真不同主题）已由 _title_similar 的
            # 「主干互为子串 + 短主干不兜底」约束住。
            if _title_similar(title, rt):
                thr = 0.0
            else:
                thr = _SECTION_MERGE_SIM
            sim = _text_sim(text, "\n".join(rlines))
            if not done and sim >= thr:
                keep = [l for l in lines
                        if not any(_text_sim(l, rl) >= 0.85 for rl in rlines)]
                if keep:
                    result[i] = (rt, rlines + keep)
                done = True
        if not done:
            result.append((title, lines))
    return result


def _merge_same_title_head(sections: list) -> list:
    """把「标题与课程同名」的小节并入紧随其后的首个小节。
    2026-09-17 新增：prompt 明确禁止「小节标题不要写成课程总标题」，但 7B 仍会
    以 `## D触发器及其边沿敏感触发器设计` 开头（等于把小节标题写成文件名），
    导致同一主题两个小节。同名节通常是承接性开头（定义/符号），并入下一节顺序不变、
    内容零丢失。只在标题长度 ≥4 且非"其他"时生效，避免误伤。

    实现用 pending 缓冲而非直接 append：命中「总标题节」后先暂存，等遇到第一个
    非总标题节再一次性落成 (下一节标题, 总标题节内容 + 下一节内容)，
    否则会先 push 一个同名条目、后续无法再并入（首版 bug）。"""
    course_title = (CFG.get("_last_course_title") or "").strip()
    if not course_title:
        return list(sections)
    out: list = []
    pending: list = []   # 待并入下一节的「总标题节」内容
    for t, lines in sections:
        is_head = bool(t) and len(t) >= 4 and t != "其他" and t == course_title
        if is_head:
            pending.extend(lines)
            continue
        if pending:
            # 落后一节：总标题节内容 + 本节内容合并到本节标题下
            out.append((t, pending + list(lines)))
            pending = []
        else:
            out.append((t, lines))
    if pending:
        # 文件末尾只有总标题节（无后继节）→ 单独成节，不丢内容
        out.append((course_title, pending))
    return out


_META_ITEM_PAT = re.compile(
    r"(课上用板书|题干在板书|未录音|"
    r"下(周|次|节|节课)[一二三四五六日]?(将|要)?(讨论|讲|学|做|上)|"
    r"为下(周|次|节)[一二三四五六日]?.{0,6}(实验|课|考试)做准备|"
    r"本节课(就)?(讲到这里|结束)|休息一下|课间休息)"
)


def _is_meta_item(line: str) -> bool:
    """判断一条目是否为课堂事务性元信息（非知识点）。"""
    return bool(_META_ITEM_PAT.search(line or ""))


def _stitch_drafts(drafts: list) -> str:
    """程序拼接多段独立草稿：同名小节内容归并 + 小节级相似归并 + 条目级去重。
    宁全勿简、内容零丢失（重复由程序消除），返回整门课完整纪要正文。"""
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
    sections = _merge_same_title_head(sections)
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
    # 2026-09-18：剔除课堂事务性元信息——7B 会把「课上用板书讲解了一道…例题
    # （题干在板书，未录音）」「介绍下周二将讨论的内容」「为下周四的实验做准备」
    # 当作知识点写进纪要。这类不是知识，进复习笔记只会占位干扰。
    # 命中任一特征即丢弃该条目；整节被剔空则连标题一起丢弃。
    sections = [(_t, [l for l in ls if not _is_meta_item(l)]) for _t, ls in sections]
    sections = [(_t, ls) for _t, ls in sections if ls]
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
    failed: list = []    # 失败段号（不阻断整体，最后汇总提示）
    for i, seg in enumerate(segments, start=1):
        t0 = time.time()
        try:
            draft = _summarize_segment(seg, meta, f"{i}/{n}", tuple(covered))
        except Exception as e:
            # 单段失败不放弃整课（2026-09-17）：重试已在 ollama_chat 内做过，
            # 到这里说明该段确实反复失败。跳过该段继续后续段，最后如实报告缺口，
            # 避免「段 5 挂了 → 前 4 段成果全丢」。
            failed.append(i)
            print(f"[纪要] 段 {i}/{n} 整理失败（已重试）：{e}", flush=True)
            continue
        drafts.append(draft)
        # 增量收集本段草稿的小节标题（任意 ##/### 层级，供后续段去重参考）
        for raw_t in re.findall(r"^#{1,4}\s+(.+?)\s*$", draft, re.M):
            t = re.sub(r"^\d+[.、)）\s]+", "", raw_t).strip()   # 去"1."序号，同主题更好匹配
            if t and len(t) <= 24 and t not in covered:
                covered.append(t)
        print(f"[纪要] 段 {i}/{n} 草稿 {len(draft)} 字符 (用时 {time.time()-t0:.0f}s)", flush=True)
    if failed:
        print(f"[纪要] 注意：第 {','.join(map(str, failed))}/{n} 段未能整理，最终纪要缺少这些段落内容", flush=True)
    if not drafts:
        raise RuntimeError(f"全部 {n} 段整理均失败，无法生成纪要")
    CFG["_last_course_title"] = (meta.get("title") or "").strip()   # 供小节标题纠偏
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
    if meta["subject"] == "待整理":
        # 2026-09-17 实证：非课程音频（90 分钟整段"点赞订阅"提示词）会被 7B 照
        # prompt 里的术语示例幻觉编造成整课纪要（伪精确比留白更毒）。
        # 宁缺毋滥：未识别出学科就不生成纪要正文，只归档转写原文。
        print("[纪要] 未识别为已知学科（内容可能不是课堂录音），跳过纪要生成，只归档转写")
        return {**meta, "summary_md": "", "fixed_text": fixed}
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


def archive(path: Path, text: str, note: dict, file_hash: str = "") -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    # 学科归一化：只允许已知学科，未知的进"待整理"（不新建学科目录）
    subject = sanitize(normalize_subject(note.get("subject")))
    title = sanitize(note.get("title") or path.stem)
    course = sanitize(note.get("course") or title)

    # 结构：学科/{原材料,纪要,转写}/ 直接放文件（文件名 = 日期_标题）
    vault = Path(CFG["vault_dir"])
    raw_dir = vault / subject / "原材料"
    note_dir = vault / subject / "纪要"
    trans_dir = vault / subject / "转写"
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

    lesson = unique_lesson(f"{date_str}_{title}", _taken)

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
        f"# {title}\n\n学科：{subject}｜课程：{course}｜日期：{date_str}\n\n{text}\n"
    )
    frontmatter = (
        f"---\n"
        f"title: {title}\n"
        f"subject: {subject}\n"
        f"course: {course}\n"
        f"date: {date_str}\n"
        f"tags: [课堂笔记, {subject}]\n"
        f"---\n\n"
    )
    summary_md = note.get("summary_md") or ""
    if not summary_md:
        summary_md = "> 未生成纪要：本课未识别为已知学科（内容可能不是课堂录音），转写原文已归档。\n"
    meta = {
        "file": audio_dst.name,
        "subject": subject,
        "course": course,
        "title": title,
        "date": date_str,
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

    print(f"[归档] -> {vault / subject}（原材料/纪要/转写）")
    return note_dst


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
        try:
            text = transcribe(str(p))
        finally:
            # 转写成功或异常都释放模型，避免下一次纪要/转写继续占用显存。
            release_whisper()
        note = generate_note(text)
        fixed = note.pop("fixed_text", text)   # 纠正后的文本给知识补全用（转写文件保留原文）
        note_path = archive(p, text, note, file_hash=h)
        if note.get("summary_md"):
            try:
                knowledge_patch(note_path, fixed, note)   # 归档后自动知识补全（7B + 维基查证）
            except Exception as e:
                # 补全失败不影响已归档纪要（设计约定），降级为警告
                print(f"[补全] 失败（纪要已归档，不影响使用）: {e}")
        else:
            print("[补全] 无纪要正文，跳过知识补全")
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
