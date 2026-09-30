# -*- coding: utf-8 -*-
"""Rebuild meeting notes from archived transcripts without re-running transcription.

Usage: python src/rerun_note.py <keyword>
The matching transcript is read from vault/会议/转写 and its paired note is updated.
"""
import sys
import io
import json
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

import core  # noqa: E402

VAULT = Path(core.CFG["vault_dir"])
TRANSCRIPTS = VAULT / "会议" / "转写"
NOTES = VAULT / "会议" / "纪要"


def find_trans(keyword: str) -> list:
    return [p for p in sorted(TRANSCRIPTS.glob("*.md"))
            if keyword in p.name or keyword in p.read_text(encoding="utf-8")[:2000]]


def rerun(transcript_path: Path) -> Path:
    text = transcript_path.read_text(encoding="utf-8")
    print(f"\n===== 重跑会议纪要: {transcript_path.name} =====", flush=True)
    started = time.time()
    note = core.generate_note(text)
    print(f"[纪要] 生成完成，用时 {time.time() - started:.0f}s", flush=True)

    note_path = NOTES / transcript_path.name
    if not note_path.exists():
        raise FileNotFoundError(f"对应纪要不存在，拒绝创建可能错配的文件：{note_path}")
    old = note_path.read_text(encoding="utf-8")
    preserved = []
    if old.startswith("---"):
        parts = old.split("---", 2)
        if len(parts) >= 3:
            preserved = [line for line in parts[1].strip().splitlines()
                         if line.startswith(("file_datetime:", "time_source:", "date:", "tags:"))]
    title = note.get("title") or "未命名会议"
    topic = note.get("topic") or "未明确"
    datetime_value = next((line.split(":", 1)[1].strip() for line in preserved
                           if line.startswith("file_datetime:")), "未记录")
    time_source = next((line.split(":", 1)[1].strip() for line in preserved
                        if line.startswith("time_source:")), "未记录")
    fm_lines = [f"title: {json.dumps(title, ensure_ascii=False)}",
                f"meeting_topic: {json.dumps(topic, ensure_ascii=False)}",
                *preserved]
    fm = "---\n" + "\n".join(fm_lines) + "\n---\n\n"
    (NOTES / ".").mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        fm + f"# {title}\n\n会议主题：{topic}｜文件时间：{datetime_value}\n"
        f"> 文件名/修改时间不一定等于会议开始时间。时间来源：{time_source}\n\n"
        + (note.get("summary_md") or ""),
        encoding="utf-8",
    )
    print(f"[写回] -> {note_path}", flush=True)
    return note_path


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1].startswith("--"):
        print(__doc__)
        sys.exit(2)
    matches = find_trans(sys.argv[1])
    if not matches:
        print(f"[无匹配] 会议转写下找不到含 '{sys.argv[1]}' 的文件")
        sys.exit(1)
    for transcript in matches:
        rerun(transcript)
    print("\n===== 全部完成 =====")
