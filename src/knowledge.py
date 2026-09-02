# -*- coding: utf-8 -*-
"""课堂笔记管家 · 知识补全模块

插在 _summarize 之后：对原文讲不全 / 录音不清的知识点做受控补全。
流程：
  1. extract_skel   本地模型提取知识点骨架 + 判定覆盖度（一次调用）
  2. patch_items    本地模型生成补全候选（一次调用，限范围）
  3. wiki_verify    存疑条目查维基百科（网络，只取摘要）
  4. finalize       基于维基摘要确认/修正补全（本地一次调用），不匹配丢弃
  5. merge_into_note 按锚点把 [补] 内容内嵌写回纪要（幂等）

范围控制铁律（写在 prompt 里）：
  - 只补"转写文本有依据"的知识点；课本后续章节内容禁止补
  - 每条补 2~4 句；不确定就标 low / needs_verify
  - 维基查不到 / 不匹配 → 丢弃，宁缺毋滥
"""
import re
import json
import time
from pathlib import Path
from difflib import SequenceMatcher

import core

# ---------------- 配置 ----------------
def _kcfg(key, default):
    return core.CFG.get("knowledge", {}).get(key, default)


MAX_GAP_ITEMS = 5      # 每篇最多补几个知识点
MAX_SENTENCES = 4      # 每条最多补几句
WIKI_LANG = "zh"       # 查证通道：维基百科语言
WIKI_TIMEOUT = 20      # 单次维基请求超时（秒）
EXCERPT_CHARS = 12000  # 喂给本地模型的转写节选长度


# ---------------- 本地模型调用 ----------------
def _chat(prompt: str, temperature: float = 0.1, max_tokens: int = 3000) -> str:
    url = core.CFG["ollama_host"] + "/api/chat"
    payload = {
        "model": core.CFG["ollama_model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_ctx": 24576,
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    r = core.requests.post(url, json=payload, timeout=900)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _extract_json(raw: str):
    """从模型输出中提取 JSON（容忍 ```json 围栏、前后杂文本、顶层对象或数组）"""
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        import json5
        return json5.loads(s)
    except Exception:
        pass
    # 提取最外层 {..} 或 [..]
    start, end = s.find("{"), s.rfind("}")
    arr_s, arr_e = s.find("["), s.rfind("]")
    if arr_s != -1 and (start == -1 or arr_s < start):
        start, end = arr_s, arr_e
    if start == -1 or end == -1:
        return None
    s = s[start:end + 1]
    try:
        import json5
        return json5.loads(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return None


# ---------------- 1. 骨架 + 覆盖度判定 ----------------
SKEL_PROMPT = """你是课程笔记审校助手。下面是{subject}《{course}》一课《{title}》的课堂录音转写文本（含听写错误，行首是时间戳）和现有 AI 纪要正文。

任务：找出"转写讲解不完整或听不清楚、且现有纪要也没有补全"的知识点，列出清单。

判定标准（关键）：
- 【先对照现有纪要】纪要中已有该知识点的标准定义/公式/完整表述 → 标 complete，不要列入清单；
- 只有"纪要里同样缺失或不完整"的知识点才需要列入：
  - partial：转写只讲了部分——只有口头描述、缺标准定义或公式、缺适用条件，且纪要里也没有补上
  - missing：转写中有线索表明老师提到或明显暗示了该知识点，但没有展开讲，纪要里也没有
  - doubtful：相关段落录音不清、听写错字连篇、表述混乱，无法判断讲了什么
- 听写错字多、口语含糊带过的，一律算 partial 或 doubtful

【严格限制】
- 只允许列转写文本中能找到依据（直接提到或明显暗示）的知识点；转写中完全没有的禁止列入
- 禁止凭学科常识、课本目录补充转写中不存在的知识点
- 每个知识点必须给出 evidence（转写中的原话片段或线索描述）

只输出 JSON，不要输出任何其他文字，格式：
{{"knowledge_points":[{{"id":1,"name":"知识点名","evidence":"原文依据","coverage":"partial|missing|doubtful|complete"}}]}}

转写文本（节选）：
{excerpt}

现有纪要正文：
{note_md}"""


def extract_skel(full_text: str, meta: dict, note_md: str = "") -> list:
    excerpt = full_text[:EXCERPT_CHARS]
    prompt = SKEL_PROMPT.format(
        subject=meta["subject"], course=meta["course"], title=meta["title"],
        excerpt=excerpt, note_md=note_md[:8000] or "(无)")
    raw = _chat(prompt)
    data = _extract_json(raw)
    if isinstance(data, list):          # 模型可能直接输出数组
        kps = data
    else:
        kps = (data or {}).get("knowledge_points") or []
    kps = [k for k in kps if isinstance(k, dict) and k.get("name")]
    print(f"[补全] 骨架 {len(kps)} 条: " + ", ".join(
        f"{k['name']}({k.get('coverage','?')})" for k in kps[:8]))
    return kps


# ---------------- 2. 补全候选（本地，限范围） ----------------
PATCH_PROMPT = """你是课程笔记补全助手。学科：{subject}｜课程：{course}｜标题：{title}

输入：① 本课录音转写文本节选；② 现有纪要正文；③ 待补知识点清单（带完整度判定）。

任务：只对下面"待补清单"中列出的知识点生成补全内容。清单之外的知识点一律禁止输出。

补全规则（严格，务必遵守）：
1. 补全内容必须给出该知识点的【标准定义、关键公式、核心定理或必要结论】——优先采用教材/课本级的标准表述；课堂没给出公式时，补全中应写出规范公式。不能只是把转写里的口语说法换个说法复述一遍；
1b. 数学公式必须用 LaTeX 写在 $...$（行内）或 $$...$$（独立一行）中，方便 Obsidian 渲染；禁止使用 \[ \] 或 \( \) 分隔符；
2. 每条补全 {max_sent} 句以内（2~4 句），宁缺毋滥；
3. 【禁止】引入待补清单之外的新知识点；禁止扩展应用、关联主题、历史沿革、习题解答；禁止补充课本后续章节内容（像勾股定理只能补本小节，不能补下一章）；
4. 【禁止】照抄或换词改写纪要正文已有内容——补全必须是纪要里缺失的增量信息（标准定义/公式/条件），与原纪要相比要有明显信息增量；
5. anchor 必须是现有纪要正文中真实存在的小节标题（## 开头）或条目名（** 开头），用于定位插入位置；纪要中没有对应位置则填"末尾"；
6. confidence 判定标准：能写出明确标准定义或公式的=high；有把握但表述不规范的=medium；需要外部资料才能确认的=low；
7. needs_verify：coverage=doubtful 或 confidence=low 时为 true；confidence=high 时必须是 false。

只输出 JSON，不要输出任何其他文字，格式：
{{"patches":[{{"id":1,"name":"知识点名","content":"补全内容","anchor":"纪要中的小节标题或条目名","confidence":"high|medium|low","needs_verify":true|false}}]}}

转写文本（节选）：
{excerpt}

现有纪要正文：
{note_md}

待补知识点清单：
{skel_json}"""


def _patched_names(note_md: str) -> list:
    """纪要中已有 [补] 行的知识点名（用于剔除已补知识点，防止重跑重复补全）"""
    names = []
    for ln in note_md.splitlines():
        s = ln.strip()
        if "[补]" not in s:
            continue
        s = s.replace("[待核验]", "").replace("[补]", "").strip()
        if s:
            names.append(s)
    return names


def patch_items(full_text: str, meta: dict, note_md: str, skel: list) -> list:
    """只对 coverage != complete 且尚未补过的条目生成补全候选，代码层二次过滤"""
    todo = [k for k in skel if k.get("coverage") != "complete"]
    if not todo:
        print("[补全] 骨架全部 complete，无需补全")
        return []
    # 剔除纪要中已有 [补] 的知识点（内容或名字匹配即视为已补）
    patched = _patched_names(note_md)
    if patched:
        def _already(nm: str) -> bool:
            for pn in patched:
                if nm and (nm in pn or pn[:20] in nm or nm in pn[:20]):
                    return True
            return False
        before = len(todo)
        todo = [k for k in todo if not _already(k.get("name", ""))]
        if len(todo) != before:
            print(f"[补全] 剔除已补知识点 {before - len(todo)} 条")
    if not todo:
        print("[补全] 待补清单均为已补/complete，无需补全")
        return []
    todo_names = {k["name"] for k in todo}
    excerpt = full_text[:EXCERPT_CHARS]
    skel_json = json.dumps(todo, ensure_ascii=False)
    prompt = PATCH_PROMPT.format(
        subject=meta["subject"], course=meta["course"], title=meta["title"],
        max_sent=MAX_SENTENCES, excerpt=excerpt, note_md=note_md[:8000], skel_json=skel_json)
    raw = _chat(prompt)
    data = _extract_json(raw)
    if isinstance(data, list):          # 模型可能直接输出数组
        patches = data
    else:
        patches = (data or {}).get("patches") or []
    patches = [p for p in patches if isinstance(p, dict) and p.get("name") and p.get("content")]
    # 硬过滤：只保留待补清单里的知识点（包含匹配，容忍名字略差异），防止越界补 complete 项
    def in_todo(name: str) -> bool:
        for tn in todo_names:
            if name in tn or tn in name:
                return True
        return False
    patches = [p for p in patches if in_todo(p["name"])]
    for p in patches:
        p.setdefault("confidence", "medium")
        p.setdefault("needs_verify", p["confidence"] == "low")
        p.setdefault("anchor", "末尾")
    # 代码层质检（7B 主力模式的防脏数据闸门）：
    #  1) 与已有 [补] 行重复（含错别字变体）→ 丢弃
    #  2) 复述纪要已有内容（无信息增量）→ 丢弃
    #  3) 复述转写原文（口语原话直接拿来，无标准表述）→ 丢弃
    patched_lines = [ln.strip() for ln in note_md.splitlines() if "[补]" in ln]
    if patched_lines:
        before = len(patches)
        patches = [p for p in patches if not _overlaps_existing(p["content"], patched_lines)]
        if len(patches) != before:
            print(f"[补全] 质检丢弃与已有补全重复 {before - len(patches)} 条")
    before = len(patches)
    patches = [p for p in patches if not _looks_like_copy(p["content"], note_md)]
    if len(patches) != before:
        print(f"[补全] 质检丢弃复述纪要 {before - len(patches)} 条")
    before = len(patches)
    patches = [p for p in patches if not (
        _looks_like_copy(p["content"], excerpt)
        or _looks_like_stitch(p["content"], excerpt))]
    if len(patches) != before:
        print(f"[补全] 质检丢弃复述转写 {before - len(patches)} 条")
    print(f"[补全] 候选 {len(patches)} 条（待补 {len(todo)} 条），需查证 "
          f"{sum(1 for p in patches if p['needs_verify'])} 条")
    return patches[:MAX_GAP_ITEMS]


def _norm(s: str) -> str:
    """压缩空白（用于内容相似度比较，消除'输出为 1'与'输出为1'这类差异）"""
    return re.sub(r"\s+", "", s)


def _looks_like_copy(content: str, note_md: str) -> bool:
    """补全内容与纪要原文重复度过高则判定为无效抄写"""
    c = content.strip()
    if len(c) < 12:
        return True
    cn, mn = _norm(c), _norm(note_md)
    # 1) 与纪要正文存在 >=15 字符连续公共片段 → 判定为复述纪要（防"换词改写已有内容"）
    m = SequenceMatcher(None, cn, mn, autojunk=False).find_longest_match(
        0, len(cn), 0, len(mn))
    if m.size >= 15:
        return True
    # 2) 长片段命中率（原逻辑保留兜底）
    segs = re.findall(r"[\u4e00-\u9fffA-Za-z0-9\[\]()=+\-*/.,，。：；、]{10,}", cn)
    if not segs:
        return False
    hit = sum(1 for s in segs if s in mn)
    return hit / len(segs) >= 0.6


def _looks_like_stitch(content: str, source: str) -> bool:
    """检测候选是否由原文多个短口语片段拼凑改写（跨行拼接口语碎片，如
    "我们列了两个方程" + "保持这个符号不变" 拼成一句伪标准表述）。
    原理：原文建 6 字连续指纹，候选非重叠命中 >=2 处即判定为拼凑复述。
    说明：只用于"复述转写"检测；正常书面标准表述几乎不会连续照抄
    口语 6 字以上片段 2 处，误杀风险低。"""
    c = content.strip()
    if len(c) < 20 or not source:
        return False
    cn, sn = _norm(c), _norm(source)
    if len(sn) < 6:
        return False
    grams = set(sn[i:i + 6] for i in range(len(sn) - 6))
    hits = 0
    last_end = -1
    for i in range(len(cn) - 6):
        if cn[i:i + 6] in grams and i >= last_end:
            hits += 1
            last_end = i + 6
    return hits >= 2


def _overlaps_existing(content: str, patched_lines: list) -> bool:
    """候选内容与纪要中已有 [补] 行共享 >=15 字符连续片段 → 判为重复补全。
    解决听写错别字（异货/异或）等名字匹配不到、但内容实际重复的情况。"""
    cn = _norm(content)
    for ln in patched_lines:
        e = _norm(ln.replace("[补]", "").replace("[待核验]", "").strip())
        if not e:
            continue
        m = SequenceMatcher(None, cn, e, autojunk=False).find_longest_match(
            0, len(cn), 0, len(e))
        if m.size >= 15:
            return True
    return False


# ---------------- 3. 维基百科查证 ----------------
_WIKI_AVAILABLE = None  # None=未探测 True/False=可达性缓存


def _wiki_endpoint(lang: str = "") -> str:
    """维基 API 端点：默认官方 zh.wikipedia.org，可在 config.json 的
    knowledge.wiki_endpoint 覆盖（如自建镜像 / 代理转发的可用地址）。"""
    ep = (_kcfg("wiki_endpoint", "") or "").strip()
    if ep:
        return ep
    return f"https://{(lang or WIKI_LANG)}.wikipedia.org/w/api.php"


def _wiki_proxies():
    """维基专用代理：config.json 的 knowledge.wiki_proxy。
    留空则不设代理（沿用系统环境变量）。"""
    wp = (_kcfg("wiki_proxy", "") or "").strip()
    if not wp:
        return None
    return {"http": wp, "https": wp}


def wiki_probe() -> bool:
    """探测维基百科当前是否可达（5s 超时），结果缓存，避免每课空等超时"""
    global _WIKI_AVAILABLE
    if _WIKI_AVAILABLE is not None:
        return _WIKI_AVAILABLE
    try:
        url = _wiki_endpoint()
        r = core.requests.get(
            url,
            params={"action": "query", "titles": "测试", "format": "json"},
            timeout=5,
            headers={"User-Agent": "class-note-keeper/1.0"},
            proxies=_wiki_proxies(),
        )
        _WIKI_AVAILABLE = r.status_code == 200
    except Exception:
        _WIKI_AVAILABLE = False
    print(f"[补全] 维基百科{'可达' if _WIKI_AVAILABLE else '不可达'}"
          f"（端点={_wiki_endpoint()}；不可达时仅写入人工定稿，7B 候选不落盘）")
    return _WIKI_AVAILABLE


def wiki_lookup(term: str, lang: str = "") -> tuple:
    """查维基百科条目摘要，返回 (title, extract) 或 (None, None)"""
    if _WIKI_AVAILABLE is False:
        return None, None
    lang = lang or WIKI_LANG
    url = _wiki_endpoint(lang)
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "format": "json",
        "redirects": 1,
        "titles": term,
    }
    headers = {"User-Agent": "class-note-keeper/1.0 (course note auto-fill)"}
    for attempt in (1, 2):
        try:
            r = core.requests.get(url, params=params, timeout=WIKI_TIMEOUT,
                                  headers=headers, proxies=_wiki_proxies())
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                if page.get("missing"):
                    return None, None
                if page.get("extract"):
                    return page.get("title"), page["extract"][:1200]
            return None, None
        except Exception as e:
            if attempt == 1:
                print(f"[补全] 维基请求失败重试: {e}")
                time.sleep(2)
            else:
                print(f"[补全] 维基请求失败: {e}")
    return None, None


# ---------------- 4. 基于维基摘要确认/修正 ----------------
VERIFY_PROMPT = """你是课程笔记校验助手。下面是几个知识点的维基百科摘要，以及对应的课堂补全候选。

对每个知识点，请判断：
- 维基摘要与课堂知识点是否指同一个概念（注意同名不同义、跨学科歧义）；
- 若匹配，则基于摘要内容对补全候选做确认或修正（2~{max_sent} 句，沿用摘要中的正确表述，仍遵守"只补本知识点、不延伸"规则）。

{results_placeholder}

只输出 JSON，不要输出任何其他文字，格式：
{{"results":[{{"id":1,"matched":true,"content":"修正后的补全内容"}},{{"id":2,"matched":false}}]}}"""


def finalize_patches(meta: dict, patches: list) -> list:
    """对全部补全候选：查维基 + 本地模型确认/修正。
    维基查得到且匹配 → 用修正后内容；查不到 → 保留本地内容但标记 verified=False（交由审校）。
    维基查到但判定不匹配 → 丢弃（防同名不同义脏数据）。"""
    if not patches:
        return patches
    blocks = []
    for p in patches:
        title, extract = wiki_lookup(p["name"])
        if extract is None:
            print(f"[补全] 维基无条目，保留待审: {p['name']}")
            p["verified"] = False
            continue
        blocks.append(
            f'### 知识点 {p["id"]}：{p["name"]}（维基条目：{title}）\n'
            f'维基摘要：{extract}\n'
            f'原补全候选：{p["content"]}\n')
    if not blocks:
        return patches

    placeholders = "\n".join(blocks)
    prompt = VERIFY_PROMPT.format(max_sent=MAX_SENTENCES, results_placeholder=placeholders)
    raw = _chat(prompt)
    data = _extract_json(raw)
    if isinstance(data, list):
        results_list = data
    else:
        results_list = (data or {}).get("results") or []
    results = {r.get("id"): r for r in results_list if isinstance(r, dict)}

    kept = []
    for p in patches:
        if p.get("verified") is False:  # 维基查不到，保留本地内容待审
            kept.append(p)
            continue
        res = results.get(p["id"])
        if not res:
            print(f"[补全] 校验无响应，保留待审: {p['name']}")
            p["verified"] = False
            kept.append(p)
            continue
        if not res.get("matched"):
            print(f"[补全] 查证不匹配，丢弃: {p['name']}")
            continue
        p["content"] = (res.get("content") or p["content"]).strip()
        p["verified"] = True
        kept.append(p)
    return kept


# ---------------- 5. 合并写回（幂等，内嵌 [补]） ----------------
def merge_into_note(note_path: Path, patches: list) -> dict:
    """把补全内容内嵌写入纪要.md。
    返回 {"inserted": n, "appended": n} 统计。"""
    note_path = Path(note_path)
    text = note_path.read_text(encoding="utf-8")
    stats = {"inserted": 0, "appended": 0, "skipped": 0}
    if not patches:
        return stats

    lines = text.splitlines()
    # 公式分隔符强制转 Obsidian 兼容格式（存量行与新内容一起转，保证幂等签名一致）
    def _obsidian_math(s: str) -> str:
        return s.replace(r"\[", "$$").replace(r"\]", "$$").replace(r"\(", "$").replace(r"\)", "$")

    lines = [_obsidian_math(ln) for ln in lines]
    # 已有的 [补] 行（用于逐条幂等：内容前缀相同的视为已插入）
    existing_patches = [ln.strip() for ln in lines if "[补]" in ln]
    for p in patches:
        name = p["name"].strip()
        content = _obsidian_math(p["content"].strip())
        if not name or not content:
            continue
        sig = content[:20]
        if any(sig in e for e in existing_patches):
            stats["skipped"] += 1
            continue
        mark = f"[补] {content}"
        anchor = (p.get("anchor") or "").strip()
        # 1) 按 anchor 精确找行
        idx = _find_line(lines, anchor)
        # 2) 按知识点名找 `- **名字` 条目
        if idx is None:
            idx = _find_name_entry(lines, name)
        # 3) 按名字找 ## 小节标题
        if idx is None:
            idx = _find_section(lines, name)
        if idx is None:
            # 追加到末尾区块
            block = f"\n- **{name}**\n  - {mark}"
            lines.append(block)
            stats["appended"] += 1
            continue
        # 找到插入点：在该行所属块末尾插入
        insert_at = _block_end(lines, idx)
        indent = "    - " if lines[idx].lstrip().startswith("- **") else "- "
        lines.insert(insert_at, f"{indent}{mark}")
        stats["inserted"] += 1

    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def _find_line(lines, anchor: str) -> int:
    if not anchor or anchor == "末尾":
        return None
    a = anchor.strip().strip("#").strip("*").strip()
    if not a:
        return None
    # 词边界匹配：锚点前后不能是中文/字母数字，避免"与或非门"误撞"与非门与或非门"这类子串
    pat = re.compile(rf"(?<![一-鿿A-Za-z0-9]){re.escape(a)}(?![一-鿿A-Za-z0-9])")
    start = _body_start(lines)
    # 优先匹配小节标题（##）与条目（- **）所在行；正文普通行作为兜底，
    # 避免"课程主题：逻辑函数与表示方式"抢在"## 逻辑函数与表示方式"之前被命中
    for priority in (0, 1):
        for i in range(start, len(lines)):
            ln = lines[i]
            s = ln.lstrip()
            if s.startswith("---") or s.startswith("|") or s.startswith("![") or s.startswith("```"):
                continue
            is_head = s.startswith("##") or s.startswith("- **")
            if (priority == 0) != is_head:
                continue
            if pat.search(ln):
                return i
    return None


def _body_start(lines) -> int:
    """返回正文起始下标（跳过 --- frontmatter 块）"""
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return i + 1
    return 0


def _find_name_entry(lines, name: str) -> int:
    start = _body_start(lines)
    for i in range(start, len(lines)):
        s = lines[i].strip()
        if s.startswith("- **") and name in s:
            return i
    return None


def _find_section(lines, name: str) -> int:
    start = _body_start(lines)
    for i in range(start, len(lines)):
        s = lines[i].strip()
        if s.startswith("##") and name in s:
            return i
    return None


def _block_end(lines, idx: int) -> int:
    """返回 idx 行所属块（条目或小节）的结束下标（插入点）"""
    cur = lines[idx]
    s0 = cur.lstrip()
    if s0.startswith("- **") or s0.startswith("- "):
        # 条目/子条目块：到下一个 - ** 条目、## 小节或 --- 为止
        # （普通子条目行也按条目块处理，避免插入点跨越到文件末尾）
        for j in range(idx + 1, len(lines)):
            s = lines[j].lstrip()
            if s.startswith("- **") or s.startswith("##") or s.startswith("---"):
                return j
        return len(lines)
    # 小节块：到下一个 ## 或 ---（frontmatter 结束）为止
    for j in range(idx + 1, len(lines)):
        s = lines[j].lstrip()
        if s.startswith("## ") or s.startswith("---"):
            return j
    return len(lines)


def _trust_local(p: dict) -> bool:
    """7B 高置信度（无需外部查证）判定：confidence=high 且明确不需查证。
    用于维基不可达 / 维基无条目时，让 7B 有把握的内容仍能写回（7B 主力模式）。"""
    return (p.get("confidence") == "high") and not p.get("needs_verify")


# ---------------- 6. 一课完整补全入口 ----------------
def patch_lesson(full_text: str, meta: dict, note_md: str, forced: list = None) -> list:
    """对一课执行完整补全流程，返回最终 patches（verified=True 的才返回）
    forced: 人工指定必补 [{"name","hint","anchor"?,"content"?}]
      - 带 content：直接采用（人工定稿，兜底，不经过 7B）
      - 不带 content：注入骨架清单，由 7B 生成（可能不被输出或质量差被丢弃）

    7B 主力模式：
      - 维基可达 → 7B 候选逐条查证，匹配的修正后写回；查无条目但 7B 高置信度的信任写回；低置信度丢弃
      - 维基不可达 → 7B 高置信度候选（confidence=high 且非 needs_verify）直接写回，低置信度丢弃
      - 人工定稿（forced 带 content）始终写回，作为兜底而非主力
    """
    t0 = time.time()
    forced = forced or []
    fixed = []
    gen_forced = []
    for f in forced:
        if not f.get("name", "").strip():
            continue
        if f.get("content"):
            fixed.append({
                "name": f["name"].strip(),
                "content": f["content"].strip(),
                "anchor": (f.get("anchor") or "末尾").strip(),
                "confidence": "high",
                "needs_verify": False,
                "verified": True,
            })
        else:
            gen_forced.append(f)
    print(f"[补全] 人工定稿 {len(fixed)} 条，需 7B 生成 {len(gen_forced)} 条")

    skel = extract_skel(full_text, meta, note_md)
    if gen_forced:
        base = 1000
        forced_skel = [
            {"id": base + i, "name": f["name"].strip(),
             "evidence": f.get("hint") or "人工指定必补", "coverage": "partial"}
            for i, f in enumerate(gen_forced)
        ]
        skel_names = {k["name"] for k in skel}
        skel = skel + [fs for fs in forced_skel if fs["name"] not in skel_names]
        print(f"[补全] 注入人工必补 {len(skel) - len(skel_names)} 条")
    if not skel:
        print("[补全] 未提取到骨架（且无人工定稿），跳过本课")
        return fixed

    wiki_ok = wiki_probe()
    patches = patch_items(full_text, meta, note_md, skel)
    if not patches:
        print("[补全] 7B 无候选")
        return fixed

    if wiki_ok:
        patches = finalize_patches(meta, patches)
        verified = []
        for p in patches:
            if p.get("verified"):
                verified.append(p)                 # 维基匹配，修正后写回
            elif _trust_local(p):
                p["verified"] = True               # 维基无条目但 7B 高置信，信任写回
                verified.append(p)
            else:
                print(f"[补全] 未通过查证且非高置信，丢弃: {p['name']}")
        print(f"[补全] 7B 候选 {len(patches)} 条，通过验证 {len(verified)} 条，"
              f"丢弃 {len(patches) - len(verified)} 条")
    else:
        # 维基不可达：7B 高置信度候选直接写回（7B 主力），低置信度宁缺毋滥
        verified = []
        for p in patches:
            if _trust_local(p):
                p["verified"] = True
                verified.append(p)
            else:
                print(f"[补全] 维基不可达且非高置信，丢弃（宁缺毋滥）: {p['name']}")
        print(f"[补全] 维基不可达，7B 高置信 {len(verified)} 条直接写回，"
              f"丢弃 {len(patches) - len(verified)} 条")
    result = fixed + verified
    # 强制项锚点：若未定位（末尾）而清单指定了 anchor，用清单锚点
    forced_anchor = {}
    for f in forced:
        fa = (f.get("anchor") or "").strip()
        if fa and fa != "末尾":
            forced_anchor[f["name"].strip()] = fa
    if forced_anchor:
        for p in result:
            pname = p["name"]
            for fname, fa in forced_anchor.items():
                if pname in fname or fname in pname:
                    if (p.get("anchor") or "").strip() in ("", "末尾"):
                        p["anchor"] = fa
                    break
    print(f"[补全] 最终补全 {len(result)} 条，用时 {time.time() - t0:.0f}s")
    return result


if __name__ == "__main__":
    # 单课测试入口：python knowledge.py <转写md> <纪要md> <学科>
    import sys
    trans_path, note_path, subject = sys.argv[1], sys.argv[2], sys.argv[3]
    full = Path(trans_path).read_text(encoding="utf-8")
    note = Path(note_path).read_text(encoding="utf-8")
    meta = {
        "subject": subject,
        "course": subject,
        "title": Path(note_path).stem,
    }
    result = patch_lesson(full, meta, note)
    print(json.dumps(result, ensure_ascii=False, indent=2))
