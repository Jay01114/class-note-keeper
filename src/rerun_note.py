# -*- coding: utf-8 -*-
"""纪要重跑工具（不重新转写）：读 vault/{学科}/转写/*.md → 新逻辑重新生成纪要 → 覆盖写回纪要/同名.md。

用途：core.py 的 SUMMARIZE_REQ / 低信息区清理等改动后，对已归档课程按新逻辑重出纪要，验证/升级质量。
whisper 零调用（转写文件已在 vault）；可选 --patch 在写回后追加知识补全（7B+维基，较慢）。

用法（用项目自带 venv，托管 python 缺 requests/json5）：
  D:/课堂笔记管家/app/.venv/Scripts/python.exe app/rerun_note.py "概率论/2026-09-07_条件独立与概率计算"
  D:/课堂笔记管家/app/.venv/Scripts/python.exe app/rerun_note.py "电路、信号和系统/叠加定理" --patch
  # 关键字匹配 vault/转写 下所有含关键字的课
"""
import sys
import io
import re
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

import core  # noqa: E402

VAULT = Path(core.CFG["vault_dir"])


def find_trans(keyword: str) -> list:
    """vault 全部转写 md 中匹配关键字的文件（不含"待整理"）"""
    hits = []
    for f in sorted(VAULT.rglob("转写/*.md")):
        if "待整理" in str(f):
            continue
        if keyword in str(f) or keyword in f.name:
            hits.append(f)
    return hits


def rerun(trans_md: Path, do_patch: bool = False) -> Path:
    """重跑单课纪要：generate_note（新逻辑）→ 覆盖写回纪要同名 md → 可选知识补全"""
    text = trans_md.read_text(encoding="utf-8")
    print(f"\n===== 重跑纪要: {trans_md.name} =====", flush=True)
    t0 = time.time()
    note = core.generate_note(text)
    print(f"[纪要] 生成完成，用时 {time.time() - t0:.0f}s", flush=True)

    # 纪要路径 = 同级目录把"转写"换"纪要"
    note_dir = trans_md.parent.parent / "纪要"
    note_path = note_dir / trans_md.name

    # 2026-09-17 新增：标题术语纠正后（如"变延敏感"→"边沿敏感"）四条文件名要一起改，
    # 否则「新标题写进旧文件名」+ 旧纪要残留在 vault（四处不同步）。
    new_title = (note.get("title") or "").strip()
    old_lesson = trans_md.stem
    if new_title and "_" in old_lesson and old_lesson[:4].isdigit():
        date_part, _, old_title = old_lesson.partition("_")
        new_lesson = f"{date_part}_{new_title}"
        if new_title != old_title:
            base = trans_md.parent.parent
            fmap = [
                (base / "转写" / f"{old_lesson}.md", base / "转写" / f"{new_lesson}.md"),
                (note_dir / f"{old_lesson}.md", note_dir / f"{new_lesson}.md"),
                (base / "原材料" / f"{old_lesson}.m4a", base / "原材料" / f"{new_lesson}.m4a"),
                (base / "原材料" / f"{old_lesson}.meta.json",
                 base / "原材料" / f"{new_lesson}.meta.json"),
            ]
            for src, dst in fmap:
                if src.exists() and not dst.exists():
                    src.rename(dst)
                    print(f"[改名] {src.name} -> {dst.name}", flush=True)
            trans_md = base / "转写" / f"{new_lesson}.md"
            note_path = note_dir / f"{new_lesson}.md"
            try:   # 转写文件首行 `# 旧标题` 一并更新，保持四处一致
                tr = trans_md.read_text(encoding="utf-8")
                tr = re.sub(r"^#\s*" + re.escape(old_title) + r"\s*$",
                            f"# {new_title}", tr, count=1, flags=re.M)
                trans_md.write_text(tr, encoding="utf-8")
            except Exception as e:
                print(f"[改名] 转写标题同步失败：{e}", flush=True)
            # meta.json 里的 title/file 同步（必须显式带 .meta.json，Path.suffix 会拆坏）
            mp = base / "原材料" / f"{new_lesson}.meta.json"
            if mp.exists():
                try:
                    import json as _json
                    mj = _json.loads(mp.read_text(encoding="utf-8"))
                    mj["title"] = new_title
                    mj["file"] = f"{new_lesson}.m4a"
                    mp.write_text(_json.dumps(mj, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as e:
                    print(f"[改名] meta.json 同步失败：{e}", flush=True)

    # 保留原 frontmatter（subject/course/date/tags），只换正文。
    # 2026-09-17 修复：旧 frontmatter 的 title 是「重跑前」的（可能含已被纠正的错字），
    # 原样保留会让新纪要顶着旧错标题 —— title 一律用本次生成的结果覆盖。
    old = note_path.read_text(encoding="utf-8") if note_path.exists() else ""
    fm_lines = []
    if old.startswith("---"):
        parts = old.split("---", 2)
        if len(parts) >= 3:
            fm_lines = [l for l in parts[1].strip().splitlines()
                        if l.startswith(("title:", "subject:", "course:", "date:", "tags:"))
                        and not l.startswith("title:")]
    real_title = new_title or (note.get("title") or "").strip()
    if real_title:
        fm_lines.insert(0, f"title: {real_title}")
    fm = "---\n" + "\n".join(fm_lines) + "\n---\n\n" if fm_lines else ""

    note_path.write_text(fm + (note.get("summary_md") or ""), encoding="utf-8")
    print(f"[写回] -> {note_path}（{len(note.get('summary_md') or '')} 字符）", flush=True)

    if do_patch:
        print("[补全] 开始知识补全（7B+维基，较慢）...", flush=True)
        core.knowledge_patch(note_path, note.get("fixed_text", text), note)
        print("[补全] 完成", flush=True)
    return note_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    keyword = sys.argv[1]
    do_patch = "--patch" in sys.argv
    files = find_trans(keyword)
    if not files:
        print(f"[无匹配] vault 转写下找不到含 '{keyword}' 的课程")
        sys.exit(1)
    for f in files:
        rerun(f, do_patch)
    print("\n===== 全部完成 =====")
