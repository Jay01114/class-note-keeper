# -*- coding: utf-8 -*-
"""Offline simulated meeting-text and archive checks. No audio/model/service is used."""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import core  # noqa: E402


class MeetingTextTests(unittest.TestCase):
    def test_metadata_is_topic_and_title_only(self):
        def fake_chat(prompt, num_predict=None):
            self.assertIn("TOPIC:", prompt)
            self.assertNotIn("SUBJECT:", prompt)
            self.assertNotIn("COURSE:", prompt)
            return "TOPIC: 项目交付时间安排\nTITLE: 项目交付安排"

        with patch.object(core, "ollama_chat", side_effect=fake_chat):
            meta = core._classify("讨论项目交付时间，以及下周复盘准备。")
        self.assertEqual(meta, {"topic": "项目交付时间安排", "title": "项目交付安排"})
        self.assertNotIn("subject", meta)
        self.assertNotIn("course", meta)

    def test_simulated_note_keeps_meeting_decisions_and_actions(self):
        outputs = iter([
            "TOPIC: 客户交付与复盘\nTITLE: 客户交付会议",
            "SUMMARY_START\n## 发生了什么事\n- 团队讨论了客户交付进度。\n"
            "## 决定与结论\n- 决定周五前发送测试包。\n"
            "## 待办事项\n- 事项：整理测试清单｜负责人：未明确｜期限：周五\n"
            "## 风险与注意事项\n- 下周一安排复盘会议。\n"
            "## 待确认问题\n- 客户侧验收人尚未确认。\nSUMMARY_END"
        ])
        with patch.object(core, "ollama_chat", side_effect=lambda *a, **k: next(outputs)):
            note = core.generate_note(
                "[00:00:01] 团队讨论客户交付进度，提议周五前发送测试包。"
                "[00:01:00] 下周一安排复盘，验收人还没确认。"
            )
        self.assertEqual(note["topic"], "客户交付与复盘")
        for expected in ("发生了什么事", "决定与结论", "待办事项", "风险与注意事项",
                         "待确认问题", "负责人：未明确", "下周一安排复盘"):
            self.assertIn(expected, note["summary_md"])
        self.assertNotIn("fixed_text", note)

    def test_stitch_preserves_similar_but_distinct_tasks_and_schedule(self):
        drafts = [
            "## 待办事项\n- 事项：发测试包｜负责人：小林｜期限：周五",
            "## 待办事项\n- 事项：发测试包｜负责人：小王｜期限：下周一\n"
            "## 风险与注意事项\n- 下周三召开复盘会",
        ]
        merged = core._stitch_drafts(drafts)
        self.assertIn("负责人：小林", merged)
        self.assertIn("负责人：小王", merged)
        self.assertIn("下周三召开复盘会", merged)

    def test_filename_time_and_mtime_fallback_are_recorded(self):
        with tempfile.TemporaryDirectory(prefix="meeting_notes_test_") as temp:
            root = Path(temp)
            vault = root / "vault"
            inbox = root / "inbox"
            inbox.mkdir()
            old_vault, old_inbox = core.CFG["vault_dir"], core.CFG["inbox_dir"]
            core.CFG["vault_dir"], core.CFG["inbox_dir"] = str(vault), str(inbox)
            try:
                audio = inbox / "recording_2026-09-30_14-25.m4a"
                audio.write_bytes(b"synthetic fixture")
                archived = core.archive(audio, "模拟原文", {
                    "topic": "项目进度协调", "title": "进度沟通",
                    "summary_md": "## 决定与结论\n- 本周完成演示",
                }, file_hash="synthetic-hash")
                self.assertEqual(archived.name, "2026-09-30_14-25_项目进度协调.md")
                meta_path = vault / "会议" / "原材料" / "2026-09-30_14-25_项目进度协调.meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.assertEqual(meta["time_source"], "filename:ymd-separated")
                self.assertFalse(meta["file_time_is_meeting_start"])
                self.assertIn("不一定等于会议开始时间", archived.read_text(encoding="utf-8"))
                self.assertTrue((vault / "会议" / "转写" / archived.name).exists())
                self.assertTrue((vault / "会议" / "原材料" / meta["file"]).exists())

                duplicate = inbox / "recording_2026-09-30_14-25.m4a"
                duplicate.write_bytes(b"second synthetic fixture")
                duplicate_note = core.archive(duplicate, "另一段模拟原文", {
                    "topic": "项目进度协调", "title": "进度沟通",
                    "summary_md": "## 待确认问题\n- 负责人需要确认",
                }, file_hash="synthetic-hash-2")
                self.assertEqual(duplicate_note.name,
                                 "2026-09-30_14-25_项目进度协调_2.md")
                self.assertIn("决定与结论",
                              archived.read_text(encoding="utf-8"))

                fallback = inbox / "voice_note.m4a"
                fallback.write_bytes(b"fallback")
                stamp = datetime(2026, 8, 7, 9, 10).timestamp()
                os.utime(fallback, (stamp, stamp))
                source_dt, source = core.source_datetime(fallback)
                self.assertEqual(source_dt.strftime("%Y-%m-%d_%H-%M"), "2026-08-07_09-10")
                self.assertEqual(source, "file_mtime")
            finally:
                core.CFG["vault_dir"], core.CFG["inbox_dir"] = old_vault, old_inbox

    def test_split_policy_remains_8000_and_1200(self):
        import inspect
        source = inspect.signature(core._split_segments)
        self.assertEqual(source.parameters["seg_chars"].default, 8000)
        self.assertEqual(source.parameters["overlap"].default, 1200)

    def test_summary_input_preserves_short_responses_and_english(self):
        source_text = (
            "[00:00:01] 对\n"
            "[00:00:02] 可以\n"
            "[00:00:03] the deployment is blocked by timeout\n"
            "[00:00:04] API 先保持 backward compatibility，周五确认。\n"
        )
        received = []

        def fake_segment(segment, meta, part):
            received.append(segment)
            return "## 发生了什么事\n- 合成测试占位事项"

        with patch.object(core, "_collapse_lowinfo",
                          side_effect=AssertionError("会议文本不应经过课堂清理")), \
             patch.object(core, "_summarize_segment", side_effect=fake_segment):
            core._summarize(source_text, {"topic": "接口部署", "title": "部署同步"})

        delivered = "\n".join(received)
        for line in source_text.splitlines():
            self.assertIn(line, delivered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
