# -*- coding: utf-8 -*-
"""课堂笔记管家 · 桌面版 V2
自设计 UI + 转写进度条 + 单实例 + 启动加速（异步拉起 Ollama）
"""
import sys
import io
import re
import os
import time
import json
import shutil
import threading
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QSharedMemory
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QPlainTextEdit,
    QHeaderView, QDialog, QFormLayout, QLineEdit, QComboBox,
    QMessageBox, QSystemTrayIcon, QMenu, QFileDialog, QAbstractItemView,
    QProgressBar, QFrame,
)

import core

APP_DIR = Path(__file__).parent
OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
OLLAMA_URL = core.CFG.get("ollama_host", "http://127.0.0.1:11434") + "/api/version"

# 转写模型档位：路径基于配置的 models_dir 拼接（可配置）
_MODELS_DIR = core.CFG.get("models_dir", "models")
WHISPER_MAP = {
    "small": str(Path(_MODELS_DIR) / "faster-whisper-small"),
    "medium": str(Path(_MODELS_DIR) / "faster-whisper-medium"),
    "large-v3": str(Path(_MODELS_DIR) / "faster-whisper-large-v3"),
}
# 转写速度倍率（秒音频 / 秒耗时），用于估算进度条
WHISPER_SPEED = {"small": 8.0, "medium": 5.0, "large-v3": 3.0}

APP_QSS = """
QMainWindow, QDialog { background: #EEF1F8; }
QWidget { font-family: "Microsoft YaHei UI", "Microsoft YaHei"; font-size: 13px; color: #2B3445; }
QLabel#appTitle { font-size: 22px; font-weight: 700; color: #3B4FE0; }
QLabel#appSub { font-size: 12px; color: #8A93A6; }
QLabel#statusTag { background: #E4EBFF; color: #3B4FE0; border-radius: 10px; padding: 3px 12px; font-weight: 600; }
QPushButton { background: #FFFFFF; border: 1px solid #D9E0EE; border-radius: 8px; padding: 7px 16px; font-weight: 600; color: #2B3445; }
QPushButton:hover { background: #F0F4FF; border-color: #3B4FE0; }
QPushButton:disabled { color: #AAB2C4; background: #F5F6FA; }
QPushButton#primaryBtn { background: #3B4FE0; color: white; border: none; }
QPushButton#primaryBtn:hover { background: #2E41C8; }
QPushButton#dangerBtn { color: #D9534F; }
QTableWidget { background: white; border: none; border-radius: 10px; gridline-color: transparent; }
QTableWidget::item { border-bottom: 1px solid #F0F2F8; padding: 6px; }
QTableWidget::item:selected { background: #E4EBFF; color: #1F2A44; }
QHeaderView::section { background: #F7F9FD; border: none; padding: 8px; font-weight: 700; color: #5A6478; }
QPlainTextEdit { background: white; border: none; border-radius: 10px; color: #4A5468; }
QProgressBar { background: #E4E9F5; border: none; border-radius: 5px; height: 10px; text-align: center; color: #2B3445; font-size: 10px; }
QProgressBar::chunk { background: #3B4FE0; border-radius: 5px; }
QFrame#card { background: white; border-radius: 12px; }
QComboBox, QLineEdit { background: white; border: 1px solid #D9E0EE; border-radius: 8px; padding: 5px 8px; }
QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar::handle:vertical { background: #C9D2E5; border-radius: 4px; min-height: 30px; }
"""


def _ollama_running() -> bool:
    try:
        import requests
        return requests.get(OLLAMA_URL, timeout=1.5).status_code == 200
    except Exception:
        return False


def _debug_log(msg: str):
    try:
        log_path = core.CONFIG_PATH.parent / "ollama_debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


class OllamaManager:
    """跟随管家启动/关闭 Ollama 服务。"""

    def __init__(self):
        self.started_by_us = False
        self.proc = None
        self._fh = None

    def ensure(self):
        """管家启动时调用：Ollama 未运行则拉起 serve（SW_HIDE 防 Windows Terminal 弹窗）"""
        if _ollama_running():
            self.started_by_us = False
            return
        if not OLLAMA_EXE.exists():
            return
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags = subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            # stdout/stderr 必须重定向到文件：windowed exe 的标准句柄无效，
            # 直接继承会让 serve 的日志/GPU 初始化行为异常（实测 7B 生成慢 3-4 倍、GPU 100%）
            _log_dir = Path(APP_DIR).parent / "tmp"
            _log_dir.mkdir(parents=True, exist_ok=True)
            _fh = open(_log_dir / "ollama_serve.log", "a", encoding="utf-8", errors="replace")
            self._fh = _fh
            self.proc = subprocess.Popen(
                [str(OLLAMA_EXE), "serve"],
                startupinfo=si,
                creationflags=subprocess.CREATE_NO_WINDOW,
                stdout=_fh,
                stderr=_fh,
            )
            self.started_by_us = True
            for _ in range(20):  # 最多等 10 秒就绪
                if _ollama_running():
                    break
                time.sleep(0.5)
        except Exception as e:
            self.started_by_us = False
            _debug_log(f"[ensure] FAILED: {e}")

    def shutdown(self):
        """管家退出时调用：只关闭管家拉起的 Ollama，不动用户自己开的"""
        if not self.started_by_us:
            return
        try:
            if self.proc:
                self.proc.terminate()
                time.sleep(1.5)
        except Exception:
            pass
        try:
            if self._fh:
                self._fh.close()
                self._fh = None
        except Exception:
            pass
        self.started_by_us = False


class LogStream(io.TextIOBase):
    """把 core 的 print 输出转成信号，供 UI 实时显示。"""

    def __init__(self, emit):
        self.emit = emit
        self.buf = ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self.emit(line)
        return len(s)

    def flush(self):
        pass


def audio_duration(path) -> float:
    """读取音频时长（秒），失败返回 0"""
    try:
        import av
        with av.open(str(path)) as container:
            if container.duration:
                return float(container.duration / av.time_base)
    except Exception:
        pass
    return 0.0


class Worker(QThread):
    log_line = Signal(str)
    task_state = Signal(str, str)      # (文件名, 状态文本，如"转写中 45%")
    task_meta = Signal(str, str, str)  # (文件名, 学科, 标题)
    task_done = Signal(str, str, float)  # (文件名, 完成/失败, 耗时秒)

    def __init__(self):
        super().__init__()
        self._stop_progress = threading.Event()
        self._manual = []               # 手动转录队列（文件名）
        self._manual_lock = threading.Lock()

    def enqueue(self, name: str):
        """手动转录入队：立即处理，绕过大小稳定检测"""
        with self._manual_lock:
            self._manual.append(name)

    def run(self):
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        sys.stdout = LogStream(self.log_line.emit)
        sys.stderr = sys.stdout
        try:
            self._run_loop()
        except Exception as e:
            self.log_line.emit(f"[致命] {e}")
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    def _run_loop(self):
        core.PROCESSED_HASHES = core.load_processed_hashes()
        self.log_line.emit(f"[去重] 已加载 {len(core.PROCESSED_HASHES)} 条历史记录")
        inbox = Path(core.CFG["inbox_dir"])
        inbox.mkdir(parents=True, exist_ok=True)
        Path(core.CFG["vault_dir"]).mkdir(parents=True, exist_ok=True)
        self.log_line.emit(f"[监视] 轮询 {inbox}，每 3 秒扫一次（关闭窗口即停止）")
        self.log_line.emit("[监视] 文件需大小稳定 6 秒才会开始处理（防止传输一半误触发）")
        seen = set()  # 只记录已开始处理的文件；启动时已有的文件走稳定检测
        stable: dict = {}
        while not core.STOP_FLAG:
            try:
                # 优先消费手动转录队列
                while True:
                    with self._manual_lock:
                        if not self._manual:
                            break
                        name = self._manual.pop(0)
                    p = inbox / name
                    if p.is_file():
                        seen.add(name)
                        self._process_one(p)
                now = time.time()
                for f in sorted(inbox.iterdir()):
                    if f.name in seen:
                        continue
                    if core.is_ready_audio(f, stable, now):
                        seen.add(f.name)
                        self._process_one(f)
                time.sleep(3)
            except Exception as e:
                self.log_line.emit(f"[监视异常] {e}")
                time.sleep(5)

    def _process_one(self, p: Path):
        name = p.name
        self.log_line.emit(f"\n===== 处理: {name} =====")
        t0 = time.time()
        try:
            h = core.file_hash(p)
            if h in core.PROCESSED_HASHES:
                dst = core.move_to(p, "_重复")
                self.log_line.emit(f"[跳过] 内容已处理过 -> {dst}")
                return
            # ---- 转写（估算进度） ----
            duration = audio_duration(p)
            model_path = core.CFG.get("whisper_model", "")
            speed = WHISPER_SPEED["medium"]
            for k, v in WHISPER_SPEED.items():
                if k in model_path:
                    speed = v
                    break
            estimate = max(duration / speed, 5.0)
            self._stop_progress.clear()

            def tick():
                while not self._stop_progress.wait(0.5):
                    elapsed = time.time() - t0
                    pct = min(int(elapsed / estimate * 100), 99) if estimate else 0
                    self.task_state.emit(name, f"转写中 {pct}%")

            pt = threading.Thread(target=tick, daemon=True)
            pt.start()
            self.log_line.emit(f"[转写] 开始: {name}")
            text = core.transcribe(str(p))
            self._stop_progress.set()
            self.log_line.emit(f"[转写] 完成（{duration:.0f}s 音频）")
            # ---- AI 纪要 ----
            self.task_state.emit(name, "AI 整理中…")
            note = core.generate_note(text)
            # ---- 归档 ----
            note_path = core.archive(p, text, note, file_hash=h)
            core.knowledge_patch(note_path, text, note)   # 归档后自动知识补全（7B + 维基查证）
            core.PROCESSED_HASHES.add(h)
            self.task_meta.emit(name, note.get("subject", ""), note.get("title", ""))
            self.task_state.emit(name, "完成")
            self.task_done.emit(name, "完成", time.time() - t0)
            self.log_line.emit(f"===== 完成: {name} =====\n")
        except Exception as e:
            self._stop_progress.set()
            self.log_line.emit(f"[失败] {name}: {e}")
            try:
                dst = core.move_to(p, "_失败")
                self.log_line.emit(f"[已移出] 避免重复处理 -> {dst}")
            except Exception:
                pass
            self.task_state.emit(name, "失败")
            self.task_done.emit(name, "失败", time.time() - t0)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(500)
        form = QFormLayout(self)
        form.setSpacing(12)

        self.inbox_edit = QLineEdit(core.CFG["inbox_dir"])
        self.vault_edit = QLineEdit(core.CFG["vault_dir"])
        self.whisper_combo = QComboBox()
        self.whisper_combo.addItem("small（最快）", "small")
        self.whisper_combo.addItem("medium（推荐，平衡）", "medium")
        self.whisper_combo.addItem("large-v3（最准最慢）", "large-v3")
        self.ollama_edit = QLineEdit(core.CFG["ollama_model"])

        cur = core.CFG["whisper_model"]
        for i in range(self.whisper_combo.count()):
            if WHISPER_MAP[self.whisper_combo.itemData(i)] == cur:
                self.whisper_combo.setCurrentIndex(i)
                break

        def pick_inbox():
            d = QFileDialog.getExistingDirectory(self, "选择收件箱目录", self.inbox_edit.text())
            if d:
                self.inbox_edit.setText(d)

        def pick_vault():
            d = QFileDialog.getExistingDirectory(self, "选择 Obsidian vault 目录", self.vault_edit.text())
            if d:
                self.vault_edit.setText(d)

        btn_inbox = QPushButton("浏览…")
        btn_inbox.clicked.connect(pick_inbox)
        btn_vault = QPushButton("浏览…")
        btn_vault.clicked.connect(pick_vault)
        box_inbox = QWidget(); h1 = QHBoxLayout(box_inbox); h1.setContentsMargins(0, 0, 0, 0)
        h1.addWidget(self.inbox_edit); h1.addWidget(btn_inbox)
        box_vault = QWidget(); h2 = QHBoxLayout(box_vault); h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(self.vault_edit); h2.addWidget(btn_vault)

        form.addRow("收件箱目录（LocalSend 导入处）", box_inbox)
        form.addRow("Obsidian vault 目录", box_vault)
        form.addRow("转写模型", self.whisper_combo)
        form.addRow("纪要模型（Ollama）", self.ollama_edit)

        btns = QHBoxLayout()
        ok = QPushButton("保存")
        ok.setObjectName("primaryBtn")
        cancel = QPushButton("取消")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btns.addStretch(); btns.addWidget(ok); btns.addWidget(cancel)
        form.addRow(btns)

    def save(self):
        cfg = core.CFG
        cfg["inbox_dir"] = self.inbox_edit.text().strip()
        cfg["vault_dir"] = self.vault_edit.text().strip()
        cfg["whisper_model"] = WHISPER_MAP[self.whisper_combo.currentData()]
        cfg["ollama_model"] = self.ollama_edit.text().strip()
        # 写入 core 实际读取的配置文件（外部优先），保证重启后仍生效
        with open(core.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        core.CFG.update(cfg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("课堂笔记管家")
        self.resize(920, 620)
        self.setMinimumSize(760, 500)
        self.worker = None
        self.tasks = {}          # 文件名 -> 行号
        self._build_ui()
        self._setup_tray()

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # 品牌区
        head = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("课堂笔记管家")
        title.setObjectName("appTitle")
        sub = QLabel("录音 → 转写 → AI 纪要 → Obsidian")
        sub.setObjectName("appSub")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        head.addLayout(title_box)
        head.addStretch()
        self.status_tag = QLabel("未启动")
        self.status_tag.setObjectName("statusTag")
        head.addWidget(self.status_tag)
        layout.addLayout(head)

        # 工具栏（卡片）
        card = QFrame()
        card.setObjectName("card")
        bar = QHBoxLayout(card)
        bar.setContentsMargins(12, 10, 12, 10)
        self.btn_start = QPushButton("开始监视")
        self.btn_start.setObjectName("primaryBtn")
        self.btn_stop = QPushButton("停止")
        self.btn_manual = QPushButton("手动转录")
        self.btn_settings = QPushButton("设置")
        self.btn_start.clicked.connect(self.start_watch)
        self.btn_stop.clicked.connect(self.stop_watch)
        self.btn_manual.clicked.connect(self.open_manual_transcribe)
        self.btn_settings.clicked.connect(self.open_settings)
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.btn_manual)
        bar.addWidget(self.btn_settings)
        bar.addStretch()
        bar.addWidget(QLabel("当前任务"))
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setValue(0)
        bar.addWidget(self.progress)
        layout.addWidget(card)

        # 任务表格（卡片）
        table_card = QFrame()
        table_card.setObjectName("card")
        tv = QVBoxLayout(table_card)
        tv.setContentsMargins(4, 6, 4, 4)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["录音文件", "学科", "课程", "状态", "耗时"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        tv.addWidget(self.table)
        layout.addWidget(table_card, 3)

        # 日志（卡片）
        log_card = QFrame()
        log_card.setObjectName("card")
        lv = QVBoxLayout(log_card)
        lv.setContentsMargins(4, 6, 4, 4)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        lv.addWidget(self.log)
        layout.addWidget(log_card, 2)

        self.setCentralWidget(central)

    # ---------- 托盘 ----------
    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.windowIcon())
        menu = QMenu()
        act_show = QAction("显示主窗口", self)
        act_show.triggered.connect(self.show_normal)
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self.quit_app)
        menu.addAction(act_show)
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self.show_normal() if reason == QSystemTrayIcon.Trigger else None
        )
        self.tray.show()

    def show_normal(self):
        self.show()
        self.setWindowState(Qt.WindowActive)

    # ---------- 控制 ----------
    def start_watch(self):
        if self.worker and self.worker.isRunning():
            self.status_tag.setText("正在停止旧任务…")
            self.tray.showMessage("课堂笔记管家", "旧任务仍在后台处理，请稍候再开始",
                                  QSystemTrayIcon.Warning, 2000)
            return
        self.tasks.clear()
        self.table.setRowCount(0)
        self.log.clear()
        self.progress.setValue(0)
        core.STOP_FLAG = False
        self.worker = Worker()
        self.worker.log_line.connect(self.on_log)
        self.worker.task_state.connect(self.on_task_state)
        self.worker.task_meta.connect(self.on_task_meta)
        self.worker.task_done.connect(self.on_task_done)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        self.status_tag.setText("监视中")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.tray.showMessage("课堂笔记管家", f"开始监视：{core.CFG['inbox_dir']}",
                              QSystemTrayIcon.Information, 3000)

    def stop_watch(self):
        """异步停止：不阻塞 UI，worker 处理完当前文件后自动退出"""
        if self.worker and self.worker.isRunning():
            core.request_stop()
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.status_tag.setText("正在停止…")
        else:
            self._on_worker_finished()

    def _on_worker_finished(self):
        self.status_tag.setText("已停止")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            dlg.save()
            self.status_tag.setText("设置已保存")
            self.tray.showMessage("课堂笔记管家", "设置已保存，下次启动生效",
                                  QSystemTrayIcon.Information, 2000)

    def open_manual_transcribe(self):
        """手动选择音频立即转录（绕过稳定检测，常用于 _失败 文件重新处理）"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择要转录的音频（可多选）", "",
            "音频文件 (*.m4a *.mp3 *.wav *.aac *.flac *.opus *.ogg *.amr *.wma *.mp4)")
        if not files:
            return
        if not (self.worker and self.worker.isRunning()):
            self.start_watch()
            if not (self.worker and self.worker.isRunning()):
                self.status_tag.setText("请稍候再试")
                return
        inbox = Path(core.CFG["inbox_dir"])
        copied = []
        for f in files:
            src = Path(f)
            dst = inbox / src.name
            if dst.exists():
                dst = inbox / f"{src.stem}_{int(time.time())}{src.suffix}"
            shutil.copy2(str(src), str(dst))
            copied.append(dst.name)
        for n in copied:
            self.worker.enqueue(n)
        self.status_tag.setText(f"已加入 {len(copied)} 个手动转录")
        self.log.appendPlainText(f"[手动] 加入转录：{', '.join(copied)}")

    # ---------- 信号处理 ----------
    def on_log(self, line):
        self.log.appendPlainText(line)

    def on_task_state(self, name, state):
        row = self._ensure_row(name)
        self.table.setItem(row, 3, QTableWidgetItem(state))
        # 全局进度条
        m = re.search(r"转写中 (\d+)%", state)
        if m:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(m.group(1)))
        elif state.startswith("AI 整理"):
            self.progress.setRange(0, 0)  # 不确定进度
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100 if state == "完成" else 0)

    def on_task_meta(self, name, subject, title):
        row = self._ensure_row(name)
        if subject:
            self.table.setItem(row, 1, QTableWidgetItem(subject))
        if title:
            self.table.setItem(row, 2, QTableWidgetItem(title))

    def on_task_done(self, name, status, seconds):
        row = self._ensure_row(name)
        self.table.setItem(row, 3, QTableWidgetItem(status))
        if status == "完成":
            self.table.setItem(row, 4, QTableWidgetItem(f"{seconds:.0f}s"))
            self.tray.showMessage("课堂笔记管家", f"{name} 处理完成", QSystemTrayIcon.Information, 3000)

    def _ensure_row(self, name) -> int:
        if name in self.tasks:
            return self.tasks[name]
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.tasks[name] = row
        self.table.setItem(row, 0, QTableWidgetItem(name))
        return row

    # ---------- 关闭 ----------
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "课堂笔记管家",
            "选择关闭方式：\n\n· 最小化到托盘 —— 继续在后台监视\n· 退出 —— 停止监视并关闭应用",
            QMessageBox.Save | QMessageBox.Close | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Save:
            event.ignore()
            self.hide()
            self.tray.showMessage("课堂笔记管家", "已最小化到托盘，继续监视",
                                  QSystemTrayIcon.Information, 2000)
        elif reply == QMessageBox.Close:
            self.quit_app()
            event.accept()
        else:
            event.ignore()

    def quit_app(self):
        self.tray.hide()
        if self.worker and self.worker.isRunning():
            core.request_stop()
            self.worker.wait(30000)  # 退出前等当前文件处理完（最多 30s）
        QApplication.quit()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("课堂笔记管家")
    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet(APP_QSS)
    app.setFont(QFont("Microsoft YaHei UI", 10))

    # 单实例：防止多点几遍拉起多个窗口
    shared = QSharedMemory("课堂笔记管家_SingleInstance_v2")
    if not shared.create(1):
        QMessageBox.information(None, "课堂笔记管家", "应用已在运行中，请查看任务栏或托盘。")
        sys.exit(0)

    # Ollama 后台拉起（不阻塞窗口显示，加快启动）
    ollama_mgr = OllamaManager()
    threading.Thread(target=ollama_mgr.ensure, daemon=True).start()
    app.aboutToQuit.connect(ollama_mgr.shutdown)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
