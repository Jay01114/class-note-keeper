# -*- coding: utf-8 -*-
"""2026-09-17 修复回归测试：archive 防覆盖 / 先移音频 / 小节归并保护 / 同名节头合并。
用项目环境运行：python tests/test_archive_merge.py"""
import sys, io, os, json, time, shutil, tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import core  # noqa: E402

PASS = FAIL = 0


def check(name, got, want):
    global PASS, FAIL
    ok = got == want
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {'✓' if ok else '✗'} {name}  got={got!r} want={want!r}")


# ---------------- 测试环境：隔离 vault ----------------
TMPROOT = Path(tempfile.mkdtemp(prefix="knk_test_"))
VAULT = TMPROOT / "vault"
INBOX = TMPROOT / "inbox"
VAULT.mkdir(parents=True)
INBOX.mkdir(parents=True)
_orig_cfg = dict(core.CFG)
core.CFG["vault_dir"] = str(VAULT)
core.CFG["inbox_dir"] = str(INBOX)
core.CFG["subjects"] = ["数字逻辑和计算机组成"]


def fake_audio(name: str, data: bytes = b"x" * 2048) -> Path:
    p = INBOX / name
    p.write_bytes(data)
    return p


def note_dict(title="测试标题"):
    return {"subject": "数字逻辑和计算机组成", "course": "数字逻辑和计算机组成",
            "title": title, "summary_md": "## 小节\n- 条目一\n- 条目二\n"}


print("\n=== ① archive 正常归档：四文件落盘 ===")
a1 = fake_audio("a1.m4a")
np1 = core.archive(a1, "转写正文一", note_dict("正常课"), file_hash="h1")
d = VAULT / "数字逻辑和计算机组成"
today = time.strftime("%Y-%m-%d")
lesson = f"{today}_正常课"
check("纪要文件存在", (d / "纪要" / f"{lesson}.md").exists(), True)
check("转写文件存在", (d / "转写" / f"{lesson}.md").exists(), True)
check("音频已移入", (d / "原材料" / f"{lesson}.m4a").exists(), True)
check("meta 存在", (d / "原材料" / f"{lesson}.meta.json").exists(), True)
check("inbox 已清空", a1.exists(), False)
mp = json.loads((d / "原材料" / f"{lesson}.meta.json").read_text(encoding="utf-8"))
check("meta.file 正确", mp["file"], f"{lesson}.m4a")

print("\n=== ② 同日同标题第二次：自动加序号，绝不覆盖 ===")
a2 = fake_audio("a2.m4a")
np2 = core.archive(a2, "转写正文二不一样", note_dict("正常课"), file_hash="h2")
lesson2 = f"{today}_正常课_2"
check("第二条用 _2 名", (d / "纪要" / f"{lesson2}.md").exists(), True)
first_txt = (d / "纪要" / f"{lesson}.md").read_text(encoding="utf-8")
check("第一条内容未被覆盖", "条目一" in first_txt, True)
check("两条转写内容不同",
      (d / "转写" / f"{lesson}.md").read_text(encoding="utf-8")
      != (d / "转写" / f"{lesson2}.md").read_text(encoding="utf-8"), True)

print("\n=== ③ 目标名被抢占 → 自动改用 _2，绝不覆盖已有数据 ===")
a3 = fake_audio("a3.m4a")
# 提前把「竞态课」的全部四个目标名占位（模拟另一课已归档）
_occ = VAULT / "数字逻辑和计算机组成"
(_occ / "纪要" / f"{today}_竞态课.md").write_text("我是别人的内容", encoding="utf-8")
(_occ / "转写" / f"{today}_竞态课.md").write_text("别人转写", encoding="utf-8")
(_occ / "原材料" / f"{today}_竞态课.m4a").write_bytes(b"other")
(_occ / "原材料" / f"{today}_竞态课.meta.json").write_text('{"file":"x"}', encoding="utf-8")
np3 = core.archive(a3, "第三条转写", note_dict("竞态课"), file_hash="h3")
check("未覆盖占用文件",
      (_occ / "纪要" / f"{today}_竞态课.md").read_text(encoding="utf-8"), "我是别人的内容")
check("未覆盖别人音频", (_occ / "原材料" / f"{today}_竞态课.m4a").read_bytes(), b"other")
check("未覆盖别人 meta",
      (_occ / "原材料" / f"{today}_竞态课.meta.json").read_text(encoding="utf-8"), '{"file":"x"}')
check("本课改用 _2 落盘", (_occ / "纪要" / f"{today}_竞态课_2.md").exists(), True)
check("_2 音频落盘", (_occ / "原材料" / f"{today}_竞态课_2.m4a").exists(), True)
_m2 = json.loads((_occ / "原材料" / f"{today}_竞态课_2.meta.json").read_text(encoding="utf-8"))
check("_2 的 meta.file 指向 _2", _m2["file"], f"{today}_竞态课_2.m4a")
check("返回值指向 _2", Path(np3).name, f"{today}_竞态课_2.md")
check("无 .part 残留", len(list(_occ.rglob("*.part"))), 0)

print("\n=== ④ 先移音频：临时文件写入失败时不把音频退回 inbox ===")
a4 = fake_audio("a4.m4a")
_orig_wt = Path.write_text
calls = {"n": 0}


def flaky_write(self, *args, **kwargs):
    if self.name.endswith(".md.part"):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("模拟写入失败")
    return _orig_wt(self, *args, **kwargs)


Path.write_text = flaky_write
try:
    try:
        core.archive(a4, "第四条", note_dict("失败课"), file_hash="h4")
        print("  ✗ 期望抛异常但没有")
        FAIL += 1
    except OSError:
        PASS += 1
        print("  ✓ 抛出了预期异常")
finally:
    Path.write_text = _orig_wt
# 关键：音频应留在 vault（不因写文件失败而回退），inbox 不再有该文件
audio_in_vault = list((d / "原材料").glob(f"{today}_失败课*.m4a"))
check("回滚后音频未留在 vault（已退回 inbox）", len(audio_in_vault), 0)
check("音频已退回 inbox", a4.exists(), True)
check("无 .part 残留", len(list(d.rglob("*.part"))), 0)
check("无半归档纪要", len(list((d / "纪要").glob(f"{today}_失败课*"))), 0)

print("\n=== ⑤ _merge_similar_sections：高相似合并 / 泛化标题保留 ===")
core._SECTION_MERGE_SIM = 0.60
near = (["- 甲" * 40] * 4)
far = ["- 完全不同内容的条目甲乙丙丁" * 12]
secs = [("D触发器的工作原理", list(near)),
        ("D触发器的工作过程", list(near)),                    # 100% 相似 → 合并
        ("泛化标题A", ["- 独有内容" * 30, "- 另一独有" * 30, "- 第三条独有" * 30]),  # 低相似 → 保留
        ("泛化标题B", ["- 别的独有内容" * 30, "- 又一条" * 30])]  # 低相似 → 保留
out = core._merge_similar_sections(secs)
check("完全相似被合并", any(t == "D触发器的工作原理" for t, _ in out), True)
check("同名节未重复", [t for t, _ in out] if False else len([t for t, _ in out]), len(out))
check("泛化标题A 保留", any(t == "泛化标题A" for t, _ in out), True)
check("泛化标题B 保留", any(t == "泛化标题B" for t, _ in out), True)
check("小节数 = 3（4→3）", len(out), 3)

print("\n=== ⑥ 单条小节不被误并（<2 条保护）===")
secs2 = [("补：时序说明", ["- 只有一条内容"]),
         ("补：时序补充", ["- 只有一条内容"])]
out2 = core._merge_similar_sections(secs2)
check("单条小节不合并", len(out2), 2)

print("\n=== ⑦ 同名小节不合并（交给 merged 阶段）===")
secs3 = [("同名标题", ["- 甲乙丙丁" * 40, "- 戊己庚辛" * 40]),
         ("同名标题", ["- 甲乙丙丁" * 40, "- 戊己庚辛" * 40])]
out3 = core._merge_similar_sections(secs3)
check("同名节保留 2 个", len(out3), 2)

print("\n=== ⑧ 标题=课程标题的小节并入下一节 ===")
core.CFG["_last_course_title"] = "D触发器及其边沿敏感触发器设计"
secs4 = [("D触发器及其边沿敏感触发器设计", ["- 定义：上升沿读取D"]),
         ("D触发器的符号表示", ["- 三角形尖端指向上升沿"]),
         ("主从结构", ["- 两个D锁存器级联"])]
out4 = core._merge_same_title_head(secs4)
check("总标题小节被吸收", len(out4), 2)
check("内容并入首节", out4[0][0], "D触发器的符号表示")
check("吸收后条目数 2", len(out4[0][1]), 2)
check("首行是原节头内容", out4[0][1][0], "- 定义：上升沿读取D")

print("\n=== ⑨ fix_terms 长词优先：变延敏感 → 边沿敏感（不出现「边沿沿」）===")
r = core.fix_terms("变延敏感除了上升延敏感，还可以有下降延敏感，也叫负延敏感",
                   "数字逻辑和计算机组成")
check("无「边沿沿」", "边沿沿" in r, False)
check("负沿敏感正确", "负沿敏感" in r, True)
check("上升沿敏感正确", "上升沿敏感" in r, True)
r2 = core.fix_terms("地出发器是边沿敏感的", "数字逻辑和计算机组成")
check("地出发器 → D触发器", "D触发器" in r2, True)
check("正确术语不被破坏", "边沿敏感" in r2, True)

print("\n=== ⑩ fix_terms 幂等：对已正确文本再跑不产生变化 ===")
good = "D触发器是边沿敏感的，上升沿敏感和负沿敏感都属于边沿触发"
check("幂等", core.fix_terms(good, "数字逻辑和计算机组成"), good)

print("\n=== ⑪ _draft_sections 识别 1~4 级标题（###/#### 不再漏成正文）===")
d1 = "## 一级节\n- 甲\n### 二级节\n- 乙\n#### 三级节\n- 丙\n# 单井号节\n- 丁\n"
secs = core._draft_sections(d1)
titles = [t for t, _ in secs]
check("识别出 4 个小节", len(secs), 4)
check("标题顺序正确", titles, ["一级节", "二级节", "三级节", "单井号节"])
check("无标题泄漏进条目",
      any("###" in l or "##" in l for _, ls in secs for l in ls), False)
check("条目数正确", sum(len(ls) for _, ls in secs), 4)

print("\n=== ⑫ 整段只用 ### 分节时不再全塞进 None 桶 ===")
d2 = "### 只有三级A\n- 内容一\n### 只有三级B\n- 内容二\n"
secs2 = core._draft_sections(d2)
check("不再产生 None 桶", any(t is None for t, _ in secs2), False)
check("两个小节都被识别", [t for t, _ in secs2], ["只有三级A", "只有三级B"])

print("\n=== ⑬ 端到端：### 草稿经 _stitch_drafts 后标题与内容不串位 ===")
draft = ("## 同步计数器设计\n- 最低位Q0在时钟上升沿翻转\n"
         "### 同步复位\n- 复位信号受时钟控制\n"
         "### 异步复位\n- 复位信号立即生效\n")
out = core._stitch_drafts([draft])
check("输出标题均为 ## 级", out.count("## "), 3)
check("无 ### 残留", "###" in out, False)
check("同步计数器设计 在内", "## 同步计数器设计" in out, True)
check("同步复位 在内", "## 同步复位" in out, True)
check("异步复位 在内", "## 异步复位" in out, True)

print("\n=== ⑭ 整段完全无标题 → 兜底切块（不再堆成一个 None 大杂烩）===")
# 注意：条目内容必须足够互异——_dedup_items 会把相似度≥0.85 的条目判为复述而合并，
# 「知识点1…知识点20」这种仅末位数字不同的假数据会彼此吃掉，属测试数据问题而非产品 bug。
_gist = ["与门实现与运算", "或门实现或运算", "非门实现取反", "与非门是与门取反",
         "或非门是或门取反", "异或门相异输出1", "同或门相同输出1", "D锁存器由门控构成",
         "SR锁存器有两个输入", "主从结构由两级级联", "上升沿在跳变瞬间采样",
         "下降沿在负跳变采样", "建立时间保证稳定", "保持时间防止误采样",
         "触发器保存一个比特", "寄存器由多个触发器组成", "计数器由触发器搭建",
         "同步计数器并行判断各位", "异步计数器逐级传递", "环形计数器首尾相接"]
d3 = "".join(f"- {g}。\n" for g in _gist)
secs3 = core._draft_sections(d3, "第2段")
check("切成 3 块（20/8 向上取整）", len(secs3), 3)
check("标题含未分节诊断信号", all("未分节" in t for t, _ in secs3), True)
check("标题可追溯到段号", all("第2段" in t for t, _ in secs3), True)
check("条目零丢失", sum(len(ls) for _, ls in secs3), 20)
out3b = core._stitch_drafts([d3])
check("拼接后条目数不丢", sum(1 for l in out3b.splitlines() if l.startswith("- ")), 20)
check("拼接出现兜底标题", "## 未分节内容" in out3b, True)

print("\n=== ⑮ 「标题→散行→标题」的散行归入前一节（不误判整段无标题）===")
d4 = "## 主从结构\n- 两个D锁存器\n- 主级跟随D\n从级在CLK=1时跟随\n## 建立时间\n- 边沿前须稳定\n"
secs4b = core._draft_sections(d4, "第1段")
check("仍是 2 个小节", len(secs4b), 2)
check("无兜底块触发", any("未分节" in t for t, _ in secs4b), False)
check("散行留在前一节内", len(secs4b[0][1]), 3)

print("\n=== ⑯ _title_similar 同义标题判定 ===")
check("A与B / A…区别 同义",
      core._title_similar("边沿敏感触发器", "边沿敏感触发器与电平敏感触发器的区别"), True)
check("A与B / B与A 语序颠倒同义",
      core._title_similar("边沿敏感触发器与电平敏感触发器的区别",
                          "电平敏感触发器与边沿敏感触发器的区别"), True)
check("子主题同义", core._title_similar("同步计数器", "同步计数器的实现"), True)
check("不同主题不误并", core._title_similar("同步计数器", "计数器的高级控制功能"), False)
check("短主干不吞并", core._title_similar("寄存器", "环形计数器与扭环形计数器"), False)
check("T/JK 不误并", core._title_similar("T触发器", "JK触发器"), False)

print("\n=== ⑰ 同义标题小节以低阈值合并（解决主题碎裂）===")
secs5 = [("边沿敏感触发器", ["- 边沿敏感按CLK边沿更新", "- 分上升沿和下降沿"]),
         ("边沿敏感触发器与电平敏感触发器的区别",
          ["- 电平敏感看电平高低决定跟随", "- 边沿敏感只在跳变瞬间读取"])]
out5 = core._merge_similar_sections(secs5)
check("两节合并为一", len(out5), 1)
check("保留先出现的标题", out5[0][0], "边沿敏感触发器")
check("条目未丢", len(out5[0][1]), 4)

print("\n=== ⑱ 课堂元信息条目不进纪要 ===")
check("板书例题被识别", core._is_meta_item(
    "- 课上用板书讲解了一道D触发器的工作原理例题（题干在板书，未录音）。"), True)
check("下周二预告被识别", core._is_meta_item(
    "- 介绍下周二将讨论的内容：寄存器和触发器的高级功能。"), True)
check("为实验做准备被识别", core._is_meta_item("- 实现目的：为下周四的实验做准备。"), True)
check("正常知识点不误伤", core._is_meta_item("- 主级在CLK=0时跟随输入D。"), False)
check("公式行不误伤", core._is_meta_item(
    "- T触发器逻辑表达式为：$Q_{n+1} = \\overline{T} \\cdot Q_n$。"), False)
d5 = ("## 例题讲解\n- 课上用板书讲解了一道例题（题干在板书，未录音）。\n"
      "## 主从结构\n- 由两个D锁存器串联。\n")
out5b = core._stitch_drafts([d5])
check("元信息被剔除", "未录音" in out5b, False)
check("整节被剔空后标题也不留", "## 例题讲解" in out5b, False)
check("正常节保留", "## 主从结构" in out5b, True)

# ---------------- 收尾 ----------------
core.CFG.clear()
core.CFG.update(_orig_cfg)
try:
    shutil.rmtree(TMPROOT, ignore_errors=True)
except Exception:
    pass

print(f"\n=== {PASS}/{PASS + FAIL} 通过 ===")
sys.exit(0 if FAIL == 0 else 1)
