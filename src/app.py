# -*- coding: utf-8 -*-
"""课堂笔记管家 · 桌面版 V3（UI 成熟化重构版）
四层信息架构：头部（状态 chip）→ 动作栏 → 任务/日志（QSplitter）→ 底部状态条。
本轮变更（2026-09-17）：
  - 视觉 token 集中化（浅色主题，颜色/字体/尺寸统一管理）
  - 持续状态（监视 chip）与瞬时反馈（3 秒回落）分离，状态用「圆点 + 文本」
  - 进度表达重做：底部状态条，6px 进度条无内嵌文字，阶段 + 百分比/不可预估
  - 关闭对话框语义重写（最小化到托盘 / 退出应用 / 取消），托盘不可用时不失联
  - 事件驱动安全退出（不在主线程 wait 冻结 UI）
  - 设置对话框防静默降档（未识别模型路径保持原值），路径校验行内错误
  - 单实例改为 QLocalServer/QLocalSocket，二次启动唤回已有窗口
  - 任务卡片（计数/操作列/空状态引导）、日志卡片（时间戳/级别/复制/自动滚动）
  - 内核修复：转写后释放显存、知识补全用纠正后文本、手动队列响应停止
"""
import sys
import io
import os
import re
import time
import json
import shutil
import html as _html
import threading
import subprocess
from pathlib import Path
from urllib.parse import quote

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QSettings
from PySide6.QtGui import QAction, QFont, QIcon, QDesktopServices, QShortcut, QKeySequence, QColor
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QPlainTextEdit,
    QTextEdit, QHeaderView, QDialog, QFormLayout, QLineEdit, QComboBox,
    QMessageBox, QSystemTrayIcon, QMenu, QFileDialog, QAbstractItemView,
    QProgressBar, QFrame, QSplitter, QStackedWidget, QCheckBox,
)

import core

APP_DIR = Path(__file__).parent
OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
# Ollama 模型目录：首次安装由 install.bat 用 setx 写入用户环境变量。
# 但 setx 只写注册表、不影响已开的会话/进程；且从资源管理器双击 exe 启动时若变量缺失，
# serve 会回退到默认目录（C 盘）→ 模型列表为 0 → 所有 /api/chat 返回 404。
# 因此这里显式兜底传入，保证无论从哪启动都能找到 D 盘模型。
OLLAMA_MODELS_DIR = Path(core.CFG.get("models_dir", str(core._APP_ROOT / "models"))) / "ollama"

APP_NAME = "课堂笔记管家"
INSTANCE_KEY = "课堂笔记管家_LocalInstance_v1"
SETTINGS_ORG = "ClassNoteKeeper"
SETTINGS_APP = APP_NAME

# 转写速度倍率（秒音频 / 秒耗时），用于估算进度条
WHISPER_SPEED = {"small": 8.0, "medium": 5.0, "large-v3": 3.0}


def whisper_speed_for(model_path: str) -> float:
    """按模型路径推导转写速度倍率（用于进度估算），未识别回落 medium。"""
    for k, v in WHISPER_SPEED.items():
        if k in (model_path or ""):
            return v
    return WHISPER_SPEED["medium"]


# ================= 视觉 token（浅色主题，集中管理） =================
C = {
    "bg": "#F5F7FB",             # 应用背景
    "card": "#FFFFFF",           # 卡片背景
    "surface": "#F7F9FD",        # 次级背景 / 表头 / hover
    "border": "#E3E8F2",         # 默认边框
    "border_strong": "#D3DAE8",  # 强边框
    "text": "#1F2A44",           # 主文字
    "text_secondary": "#6B7688", # 次文字
    "brand": "#3B4FE0",
    "brand_hover": "#2E41C8",
    "brand_pressed": "#2537AD",
    "brand_light": "#E4EBFF",
    "brand_disabled": "#C7CEF6",
    "success": "#1F9D55",
    "success_light": "#E7F6EE",
    "warning": "#B26A00",
    "warning_light": "#FFF4E0",
    "danger": "#D64545",
    "danger_light": "#FDECEC",
    "focus": "#7A88EA",
}
SPACING = (4, 8, 12, 16, 20, 24, 32)
CTRL_H = 36          # 控件高
ROW_H = 40           # 表格行高
RADIUS_CTRL = 6
RADIUS_CARD = 10

APP_QSS = f"""
QMainWindow, QDialog {{ background: {C['bg']}; }}
QWidget {{ font-family: "Microsoft YaHei", "Microsoft YaHei UI"; font-size: 14px; color: {C['text']}; }}

QLabel#appTitle {{ font-size: 20px; font-weight: 600; color: {C['text']}; }}
QLabel#appSub {{ font-size: 12px; color: {C['text_secondary']}; }}
QLabel#cardTitle {{ font-size: 16px; font-weight: 600; color: {C['text']}; }}
QLabel#hint {{ font-size: 12px; color: {C['text_secondary']}; }}
QLabel#inlineError {{ font-size: 12px; color: {C['danger']}; }}
QLabel#chip {{ background: {C['surface']}; border: 1px solid {C['border']};
    border-radius: 11px; padding: 4px 12px; font-size: 12px; font-weight: 600; color: {C['text']}; }}

QPushButton {{ background: {C['card']}; border: 1px solid {C['border_strong']};
    border-radius: {RADIUS_CTRL}px; padding: 0 16px; min-height: {CTRL_H - 2}px;
    font-size: 14px; font-weight: 700; color: {C['text']}; }}
QPushButton:hover {{ background: {C['surface']}; border-color: {C['brand']}; }}
QPushButton:pressed {{ background: {C['brand_light']}; }}
QPushButton:focus {{ border: 1px solid {C['focus']}; }}
QPushButton:disabled {{ color: #AAB2C4; background: {C['surface']}; border-color: {C['border']}; }}
QPushButton#primaryBtn {{ background: {C['brand']}; color: #FFFFFF; border: 1px solid {C['brand']}; }}
QPushButton#primaryBtn:hover {{ background: {C['brand_hover']}; border-color: {C['brand_hover']}; }}
QPushButton#primaryBtn:pressed {{ background: {C['brand_pressed']}; border-color: {C['brand_pressed']}; }}
QPushButton#primaryBtn:focus {{ border: 1px solid {C['focus']}; }}
QPushButton#primaryBtn:disabled {{ background: {C['brand_disabled']}; color: #F2F4FF;
    border-color: {C['brand_disabled']}; }}
QPushButton#dangerTextBtn {{ color: {C['danger']}; }}
QPushButton[flat="true"] {{ border: none; background: transparent; color: {C['brand']};
    min-height: 24px; padding: 0 8px; }}
QPushButton[flat="true"]:hover {{ color: {C['brand_hover']}; background: {C['brand_light']}; }}
QPushButton[flat="true"]:disabled {{ color: #AAB2C4; background: transparent; }}

QFrame#card {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: {RADIUS_CARD}px; }}
QFrame#statusBar {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: {RADIUS_CARD}px; }}

QTableWidget {{ background: {C['card']}; border: none; gridline-color: transparent; }}
QTableWidget::item {{ border-bottom: 1px solid #F0F2F8; padding: 4px 8px; }}
QTableWidget::item:selected {{ background: {C['brand_light']}; color: {C['text']}; }}
QHeaderView::section {{ background: {C['surface']}; border: none;
    border-bottom: 1px solid {C['border']}; padding: 8px; font-weight: 600; color: {C['text_secondary']}; }}

QTextEdit#logView {{ background: {C['card']}; border: none;
    font-family: "Cascadia Mono", "Consolas"; font-size: 12px; color: {C['text']}; }}
QPlainTextEdit {{ background: {C['card']}; border: none; }}

QLineEdit, QComboBox {{ background: {C['card']}; border: 1px solid {C['border_strong']};
    border-radius: {RADIUS_CTRL}px; padding: 0 8px; min-height: {CTRL_H - 2}px; }}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {C['focus']}; }}
QLineEdit[invalid="true"] {{ border: 1px solid {C['danger']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {C['card']}; border: 1px solid {C['border']};
    selection-background-color: {C['brand_light']}; selection-color: {C['text']}; }}

QProgressBar {{ background: #E4E9F5; border: none; border-radius: 3px;
    min-height: 6px; max-height: 6px; }}
QProgressBar::chunk {{ background: {C['brand']}; border-radius: 3px; }}
QProgressBar[danger="true"]::chunk {{ background: {C['danger']}; }}

QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {C['border_strong']};
    border-radius: 3px; background: {C['card']}; }}
QCheckBox::indicator:checked {{ background: {C['brand']}; border-color: {C['brand']};
    image: none; }}

QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #C9D2E5; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #B4BFD6; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #C9D2E5; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QSplitter::handle {{ background: {C['bg']}; }}
QSplitter::handle:vertical {{ height: 8px; }}

QToolTip {{ background: {C['text']}; color: #FFFFFF; border: none; padding: 6px 8px; }}
"""


# ================= AI 引擎探测 =================
ENGINE_STARTING = "启动中"
ENGINE_READY = "就绪"
ENGINE_READY_CPU = "就绪 · CPU"
ENGINE_MODEL_MISSING = "模型缺失"
ENGINE_UNAVAILABLE = "不可用"

# GPU 发现是否成功（2026-09-18）。_wait_gpu_ready 写入，EngineProbe 读取后
# 把状态显示成「就绪 · CPU」——CPU 模式下 7B 慢 3-4 倍、上下文被压到 4096，
# 纪要质量会明显下滑，必须让用户一眼看见，而不是只留在日志里。
_ENGINE_GPU_OK = True


def _ollama_running() -> bool:
    try:
        import requests
        host = core.CFG.get("ollama_host", "http://127.0.0.1:11434")
        return requests.get(host + "/api/version", timeout=1.5).status_code == 200
    except Exception:
        return False


def _ollama_has_model(model: str = None) -> bool:
    """确认 serve 真的加载到了目标模型。
    仅探测端口存活不够：若 OLLAMA_MODELS 指向错误，serve 正常响应但模型列表为空，
    之后每个课时的 /api/chat 都会 404 白白浪费一整轮转写。"""
    model = model or core.CFG.get("ollama_model", "qwen2.5:7b")
    try:
        import requests
        host = core.CFG.get("ollama_host", "http://127.0.0.1:11434")
        r = requests.get(host + "/api/tags", timeout=2.5)
        if r.status_code != 200:
            return False
        names = [m.get("name", "") for m in r.json().get("models", [])]
        base = model.split(":")[0]
        return any(n == model or n.split(":")[0] == base for n in names)
    except Exception:
        return False


def _wait_gpu_ready(log_path, start_offset: int = None, timeout: float = 45.0) -> bool:
    """等 Ollama 完成 GPU discovery 再放行（2026-09-17 内核修复）。

    背景：serve 的 /api/version 端口先就绪，GPU discovery 仍在后台跑（实测约
    20 秒）。这段窗口里发起的 /api/chat 会被路由到 CPU runner：
      · 日志出现 "inference compute ... library=cpu" 且 total_vram="0 B"
      · 7B 生成慢 3-4 倍，且 vram-based default context 退化成 4096
    结果就是纪要质量整体塌方，而 UI 上只表现为「变慢」，极难归因。

    判定成功的标志是日志里出现带 library=CUDA 的 inference compute 行。
    start_offset 由调用方在「拉起 serve 之前」取日志文件大小，保证只解析
    本次启动新写入的内容——否则会命中上一轮失败留下的 library=cpu 行而误判。
    超时返回 False 但不抛异常（退回旧行为，只影响是否走 GPU，不影响可用性）。
    """
    try:
        import re as _re
        p = Path(log_path)
        if not p.exists():
            return False
        if start_offset is None:
            start_offset = p.stat().st_size
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(max(0, start_offset))
                    tail = f.read()
            except Exception:
                tail = ""
            if _re.search(r"inference compute.*library=CUDA", tail):
                _debug_log("[ensure] GPU discovery 完成（library=CUDA），引擎走 GPU")
                globals()["_ENGINE_GPU_OK"] = True
                return True
            if _re.search(r"inference compute.*library=cpu", tail):
                # 已落到 CPU 分支：同一次启动内不会二次发现，立即放行
                _debug_log("[ensure] 警告：Ollama 落到 CPU runner（GPU discovery 失败）")
                globals()["_ENGINE_GPU_OK"] = False
                return False
            time.sleep(1.0)
        _debug_log("[ensure] 等待 GPU discovery 超时，继续（不阻塞启动）")
        globals()["_ENGINE_GPU_OK"] = False
        return False
    except Exception as e:
        _debug_log(f"[ensure] _wait_gpu_ready 异常: {e}")
        return False


class EngineProbe(QThread):
    """异步探测 AI 引擎状态（不阻塞 UI）。probed 发出 ENGINE_* 常量。

    grace_seconds：宽限期。Ollama 冷启动实测需要 5 秒以上（2026-09-17 实证：
    应用 09:09:45 启动、ollama 09:09:50 才拉起），此前启动 3s 探测一次即判
    「不可用」且永不重试，chip 卡死红色。现改为：宽限期内每 interval 秒重试、
    状态保持「启动中」；超过宽限期仍连不上才判「不可用」。
    """
    probed = Signal(str)

    def __init__(self, grace_seconds: float = 0.0, interval: float = 2.0, parent=None):
        super().__init__(parent)
        self._grace = float(grace_seconds)
        self._interval = max(float(interval), 0.5)
        self._stop = False

    def stop(self):
        """优雅停止：run 循环检测到后尽快退出，配合 wait() 防止线程未结束就被析构
        （2026-09-17 根因：QThread 运行中析构 → Qt 6.11 fail-fast → BEX64 静默闪退）"""
        self._stop = True

    def run(self):
        if self._grace > 0:
            self.probed.emit(ENGINE_STARTING)
        deadline = time.time() + self._grace
        while True:
            if self._stop:
                return
            if _ollama_running():
                ready = _ollama_has_model()
                if ready and not _ENGINE_GPU_OK:
                    self.probed.emit(ENGINE_READY_CPU)
                else:
                    self.probed.emit(ENGINE_READY if ready else ENGINE_MODEL_MISSING)
                return
            if time.time() >= deadline:
                self.probed.emit(ENGINE_UNAVAILABLE)
                return
            time.sleep(self._interval)


def _debug_log(msg: str):
    """调试日志写产品根目录下的 tmp/（不再是硬编码 D 盘路径，开源分发可移植）。
    根目录 tmp 不可写时静默跳过——调试日志绝不能影响主流程。"""
    try:
        _d = core._APP_ROOT / "tmp"
        _d.mkdir(parents=True, exist_ok=True)
        with open(_d / "ollama_debug.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


class OllamaManager:
    """跟随管家启动/关闭 Ollama 服务。"""

    def __init__(self):
        self.started_by_us = False
        self.proc = None
        self._fh = None

    def _spawn(self, log_file: Path) -> int:
        """以「脱离父进程」的方式拉起 ollama serve，返回本次日志写入起点偏移。

        2026-09-18 修复（纪要质量塌方的真凶之一）：
        实测应用（PyInstaller windowed exe）拉起的 ollama，其 GPU 发现会对
        cuda_v12 / cuda_v13 / rocm / vulkan **全部崩溃**（llama-server
        --list-devices 返回 0xc0000005），最终退到 CPU runner：
          inference compute ... library=cpu + total_vram="0 B" + default_num_ctx=4096
        而同机、同环境变量、同 PATH、同 CWD 下用普通进程手动起 ollama，
        GPU 发现正常（library=CUDA compute=12.0, total_vram=7.9 GiB）。
        已逐项排除：环境变量（仅 NO_PROXY 差异）、PATH（含 _internal 前置）、
        CWD、stdout/stderr 重定向、stdin 继承 → 结论是**进程上下文继承**问题
        （windowed exe 的控制台/句柄/JOB 上下文被子进程继承）。

        对策：**保持 CREATE_NO_WINDOW**（给 serve 一个隐藏控制台，其派生的
        llama-server / runner 继承该控制台 → 全程不弹黑窗），stdin 显式接
        DEVNULL、close_fds 显式开启，日志重定向到文件。

        ⚠️ 别改成 DETACHED_PROCESS：2026-09-18 实测那样会让 ollama 没有控制台，
        它派生的每个子进程各自新建控制台 → Win11 上弹 30+ 个 Windows Terminal 黑窗。
        （GPU 退 CPU 的真因是 `_internal\msvcp140.dll` 版本漂移，与本处无关。）
        """
        try:
            _start = log_file.stat().st_size
        except Exception:
            _start = 0
        env = os.environ.copy()
        # 显式带上 OLLAMA_MODELS：不依赖用户环境变量（setx 不刷新当前会话；
        # 且双击 exe 启动时可能未继承）→ 防止 serve 回退默认目录、模型列表为 0
        env["OLLAMA_MODELS"] = str(OLLAMA_MODELS_DIR)
        si = subprocess.STARTUPINFO()
        si.dwFlags = subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        self._fh = open(log_file, "a", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            [str(OLLAMA_EXE), "serve"],
            startupinfo=si,
            # ⚠️ 必须是 CREATE_NO_WINDOW，不能改 DETACHED_PROCESS！
            # 2026-09-18 实测：DETACHED_PROCESS 让 ollama 完全没有控制台，
            # 于是它派生的每个 llama-server / runner 都被分配一个**新的控制台**，
            # 而 Windows 11 默认终端是 Windows Terminal →
            # 启动一次应用弹出 30+ 个黑窗（50s 抓拍：WindowsTerminal 17 个、
            # llama-server 11 个、ollama 6 个）。
            # CREATE_NO_WINDOW 会给 ollama 一个**隐藏的控制台**，派生的子进程继承它，
            # 全程零窗口（这本就是 v15 的既有行为，不要动）。
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL,
            stdout=self._fh,
            stderr=self._fh,
            close_fds=True,
            env=env,
        )
        self.started_by_us = True
        return _start

    def _kill_ours(self, wait: float = 2.0):
        """杀掉本次拉起的 serve（重试用），并关掉日志句柄。"""
        try:
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass
        time.sleep(wait)
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        try:
            if self._fh:
                self._fh.close()
                self._fh = None
        except Exception:
            pass
        self.proc = None
        self.started_by_us = False

    def ensure(self):
        """管家启动时调用：Ollama 未运行则拉起 serve 并确认走 GPU。

        GPU 发现失败时**自动重启一次**再试：实测失败是进程上下文引起的偶发
        崩溃（0xc0000005），干净重启即可恢复为 CUDA。仍失败才退回 CPU，
        并把「跑在 CPU 上」写进日志（CPU 模式下 7B 慢 3-4 倍、上下文被压到
        4096，纪要质量会明显下滑——必须留痕，不能让用户只看到「变慢」）。
        """
        if _ollama_running():
            self.started_by_us = False
            return
        if not OLLAMA_EXE.exists():
            return
        try:
            if getattr(sys, "frozen", False):
                # dist\课堂笔记管家\课堂笔记管家.exe → 项目根
                _root = Path(sys.executable).parent.parent.parent
            else:
                _root = Path(APP_DIR).parent
            _log_dir = _root / "tmp"
            _log_dir.mkdir(parents=True, exist_ok=True)
            _log_file = _log_dir / "ollama_serve.log"

            for attempt in (1, 2):
                _log_start = self._spawn(_log_file)
                for _ in range(40):          # 最多等 20 秒端口就绪
                    if _ollama_running():
                        break
                    time.sleep(0.5)
                # 端口就绪 ≠ 引擎可用：GPU discovery 仍要在后台跑（实测约 20 秒）。
                # 这段窗口里 /api/chat 会被路由到 CPU runner，7B 慢 3-4 倍且
                # num_ctx 被压到 4096 → 纪要质量塌方。
                if _wait_gpu_ready(_log_file, _log_start, timeout=45.0):
                    return
                if attempt == 1:
                    _debug_log("[ensure] GPU 发现失败，重启 Ollama 重试一次 …")
                    self._kill_ours()
                    time.sleep(3.0)
                    continue
                _debug_log("[ensure] 重试后仍未走 GPU：本次会话将使用 CPU 运行，"
                           "纪要质量可能下降（请检查显卡驱动或见 ollama_serve.log）")
        except Exception as e:
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


# ================= UI 状态模型（纯逻辑，便于单测） =================
WATCH_NOT_STARTED = "未启动"
WATCH_WATCHING = "监视中"
WATCH_STOPPING = "正在停止"
WATCH_STOPPED = "已停止"


def next_watch_state(cur: str, event: str) -> str:
    """持续状态机（纯函数，单测覆盖）。
    事件：start / stopping / stopped / save / manual。
    规则：save、manual 是瞬时事件，不得改变持续状态——
    「监视中」时保存设置或手动入队，chip 必须仍显示「监视中」。"""
    if event in ("save", "manual"):
        return cur
    if event == "start":
        return WATCH_WATCHING
    if event == "stopping":
        return WATCH_STOPPING
    if event == "stopped":
        return WATCH_STOPPED
    return cur


def whisper_options(models_dir: str):
    """由 models_dir 推导可用转写模型（不写死项目绝对路径）。
    扫描 faster-whisper-* 目录，返回 [(名称, 路径)]，按约定顺序排列。"""
    base = Path(models_dir)
    found = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.is_dir() and d.name.startswith("faster-whisper-"):
                found.append((d.name[len("faster-whisper-"):], d.resolve().as_posix()))
    order = ["small", "medium", "large-v3", "large-v2", "large"]
    found.sort(key=lambda kv: (order.index(kv[0]) if kv[0] in order else 99, kv[0]))
    return found


def whisper_combo_entries(models_dir: str, current: str):
    """设置下拉框选项：返回 (items, current_index)，items=[(label, value)]。
    当前配置值无法与任何模型目录匹配时，首项显示「当前配置（未识别）：路径」
    并保持原值——绝不静默改成 small（防止用户只点保存就降档）。"""
    items = [(f"{n}", v) for n, v in whisper_options(models_dir)]
    norm = (current or "").replace("\\", "/")
    cur_idx = -1
    for i, (_, v) in enumerate(items):
        if norm == v.replace("\\", "/"):
            cur_idx = i
            break
    if cur_idx < 0:
        items.insert(0, (f"当前配置（未识别）：{current}", current))
        cur_idx = 0
    return items, cur_idx


def unique_dst_name(inbox: Path, name: str, reserved=None) -> Path:
    """重名文件用可读后缀 (2)、(3)…。

    reserved 用于手动批次规划阶段，避免同一批次的两个同名源文件得到同一个目标。
    """
    if reserved is None:
        reserved = set()
    dst = inbox / name
    if not dst.exists() and str(dst).casefold() not in reserved:
        return dst
    stem, suf = Path(name).stem, Path(name).suffix
    n = 2
    while True:
        candidate = inbox / f"{stem} ({n}){suf}"
        if not candidate.exists() and str(candidate).casefold() not in reserved:
            return candidate
        n += 1


# ================= 后台处理线程 =================
class Worker(QThread):
    log_line = Signal(str)                      # 日志行
    task_state = Signal(str, str)               # (文件名, 状态文本)
    task_meta = Signal(str, str, str)           # (文件名, 学科, 标题)
    task_done = Signal(str, str, float, str)    # (文件名, 终态, 耗时秒, 笔记目录)
    progress = Signal(str, str, int)            # (文件名, 阶段, 百分比；-1=不可预估)

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
        seen = set()  # 只记录正在处理的文件；处理完即清理（同名新文件重新走稳定检测）
        stable: dict = {}
        while not core.STOP_FLAG:
            try:
                # 优先消费手动转录队列：每个文件之间检查停止请求（2026-09-17 修复）
                while True:
                    if core.STOP_FLAG:
                        with self._manual_lock:
                            remaining, self._manual = self._manual[:], []
                        for name in remaining:
                            self.log_line.emit(f"[取消] 已停止监视，未处理：{name}")
                            self.task_state.emit(name, "已取消")
                            self.task_done.emit(name, "已取消", 0.0, "")
                        break
                    with self._manual_lock:
                        if not self._manual:
                            break
                        name = self._manual.pop(0)
                    p = inbox / name
                    if p.is_file():
                        seen.add(name)
                        self._process_one(p)
                        core.seen_cleanup(seen, stable, name)
                    else:
                        self.log_line.emit(f"[失败] {name}: 文件不在收件箱中")
                        self.task_state.emit(name, "失败")
                        self.task_done.emit(name, "失败", 0.0, "")
                if core.STOP_FLAG:
                    break
                now = time.time()
                for f in sorted(inbox.iterdir()):
                    if core.STOP_FLAG:
                        break
                    if f.name in seen:
                        continue
                    if core.is_ready_audio(f, stable, now):
                        seen.add(f.name)
                        self._process_one(f)
                        core.seen_cleanup(seen, stable, f.name)
                time.sleep(3)
            except Exception as e:
                self.log_line.emit(f"[监视异常] {e}")
                time.sleep(5)

    def _process_one(self, p: Path):
        name = p.name
        self.log_line.emit(f"===== 处理: {name} =====")
        t0 = time.time()
        try:
            h = core.file_hash(p)
            if h in core.PROCESSED_HASHES:
                dst = core.move_to(p, "_重复")
                self.log_line.emit(f"[跳过] 内容已处理过 -> {dst}")
                self.task_state.emit(name, "已跳过（重复）")
                self.progress.emit(name, "完成", 100)
                self.task_done.emit(name, "已跳过（重复）", time.time() - t0, "")
                return
            # ---- 转写（估算进度） ----
            duration = audio_duration(p)
            model_path = core.CFG.get("whisper_model", "")
            speed = whisper_speed_for(model_path)
            estimate = max(duration / speed, 5.0)
            self._stop_progress.clear()
            self.progress.emit(name, "转写", 0)

            def tick():
                while not self._stop_progress.wait(0.5):
                    elapsed = time.time() - t0
                    pct = min(int(elapsed / estimate * 100), 99) if estimate else 0
                    self.task_state.emit(name, f"转写中 {pct}%")
                    self.progress.emit(name, "转写", pct)

            pt = threading.Thread(target=tick, daemon=True)
            pt.start()
            self.log_line.emit(f"[转写] 开始: {name}")
            try:
                text = core.transcribe(str(p))
            finally:
                # 内核修复（2026-09-17）：转写后立即释放显存，7B 纪要阶段独占 GPU。
                # large-v3 + 7B 同驻超 8GB 会卡死；异常路径同样释放（try/finally）。
                # 2026-09-17 二次修复：release_whisper 自身异常必须吞掉——它写在
                # finally 里，一旦抛出会顶掉 transcribe 的真实异常，让上层只看到
                # 一个无关的释放错误（掩盖根因）。
                self._stop_progress.set()
                try:
                    core.release_whisper()
                except Exception as e:
                    self.log_line.emit(f"[转写] 释放显存失败（忽略）: {e}")
            self.log_line.emit(f"[转写] 完成（{duration:.0f}s 音频）")
            # ---- AI 纪要 ----
            self.task_state.emit(name, "AI 整理中")
            self.progress.emit(name, "AI 整理", -1)   # 不可预估
            note = core.generate_note(text)
            # 内核修复（2026-09-17）：知识补全必须用术语纠正后的 fixed_text（与 CLI 一致）；
            # 转写文件仍保存原始 text。
            fixed = note.pop("fixed_text", text)
            # ---- 归档 ----
            self.task_state.emit(name, "归档中")
            self.progress.emit(name, "归档", -1)
            note_path = core.archive(p, text, note, file_hash=h)
            # ---- 知识补全 ----
            if note.get("summary_md"):
                self.task_state.emit(name, "知识补全中")
                self.progress.emit(name, "知识补全", -1)
                try:
                    core.knowledge_patch(note_path, fixed, note)
                except Exception as e:
                    # 补全失败不影响已归档纪要（设计约定），降级为警告
                    self.log_line.emit(f"[补全] 失败（纪要已归档，不影响使用）: {e}")
            else:
                self.log_line.emit("[补全] 无纪要正文（未识别学科），跳过知识补全")
            core.PROCESSED_HASHES.add(h)
            self.task_meta.emit(name, note.get("subject", ""), note.get("title", ""))
            self.task_state.emit(name, "完成")
            self.progress.emit(name, "完成", 100)
            self.task_done.emit(name, "完成", time.time() - t0, str(note_path.parent))
            self.log_line.emit(f"===== 完成: {name} =====")
        except Exception as e:
            self._stop_progress.set()
            try:
                core.release_whisper()   # 异常路径同样保证释放转写显存
            except Exception:
                pass
            self.log_line.emit(f"[失败] {name}: {e}")
            try:
                dst = core.move_to(p, "_失败")
                self.log_line.emit(f"[已移出] 避免重复处理 -> {dst}")
            except Exception:
                pass
            self.task_state.emit(name, "失败")
            self.task_done.emit(name, "失败", time.time() - t0, "")


# ================= 日志卡片 =================
def log_level_of(text: str) -> str:
    """识别 core 日志前缀 → UI 层级别（不改 core 的日志协议）。"""
    if any(t in text for t in ("[失败]", "[致命]", "[监视异常]")):
        return "danger"
    if any(t in text for t in ("[跳过]", "[取消]")):
        return "warning"
    if "[补全]" in text:
        return "success"
    if "[转写]" in text:
        return "brand"
    if any(t in text for t in ("[监视]", "[去重]", "[手动]", "[纪要]", "[归档]")):
        return "secondary"
    return "normal"


LOG_LINE_COLOR = {
    "danger": C["danger"],
    "warning": C["warning"],
    "success": C["success"],
    "brand": C["brand"],
    "secondary": C["text_secondary"],
    "normal": C["text"],
}

LOG_FILTERS = [("全部", None), ("失败", "danger"), ("跳过", "warning"),
               ("监视", "secondary"), ("转写", "brand"), ("补全", "success")]


class LogView(QWidget):
    """日志卡片：时间戳 + 级别着色 + 级别筛选 + 复制 + 自动滚动。
    等宽字体；用户向上滚暂停自动滚动，回到底部恢复。"""
    MAX_ENTRIES = 1500

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        title = QLabel("日志")
        title.setObjectName("cardTitle")
        head.addWidget(title)
        self.filter_combo = QComboBox()
        for label, _ in LOG_FILTERS:
            self.filter_combo.addItem(label)
        self.filter_combo.setFixedWidth(90)
        self.filter_combo.currentIndexChanged.connect(lambda _: self._render_all())
        head.addWidget(self.filter_combo)
        head.addStretch()

        btn_copy_sel = QPushButton("复制选中")
        btn_copy_sel.setProperty("flat", True)
        btn_copy_sel.clicked.connect(self._copy_selected)
        btn_copy_all = QPushButton("复制全部")
        btn_copy_all.setProperty("flat", True)
        btn_copy_all.clicked.connect(self._copy_all)
        btn_clear = QPushButton("清空显示")
        btn_clear.setProperty("flat", True)
        btn_clear.clicked.connect(self.clear)
        for b in (btn_copy_sel, btn_copy_all, btn_clear):
            head.addWidget(b)
        self.autoscroll_chk = QCheckBox("自动滚动")
        self.autoscroll_chk.setChecked(True)
        self.autoscroll_chk.toggled.connect(self._on_autoscroll_toggled)
        head.addWidget(self.autoscroll_chk)
        v.addLayout(head)

        self.view = QTextEdit()
        self.view.setObjectName("logView")
        self.view.setReadOnly(True)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sb = self.view.verticalScrollBar()
        sb.valueChanged.connect(self._on_scroll)
        v.addWidget(self.view)

        self._entries = []   # (timestamp, level, text, tag)

    # ---- 数据 ----
    def append_line(self, text: str):
        tag = ""
        if text.startswith("===== 处理: ") and text.endswith(" ====="):
            tag = text[len("===== 处理: "):-len(" =====")]
        # 时间必须在日志产生时记录；重新筛选/重绘时不能重新取当前时间。
        entry = (time.strftime("%H:%M:%S"), log_level_of(text), text, tag)
        self._entries.append(entry)
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[-self.MAX_ENTRIES:]
            self._render_all()
            return
        if self._passes(entry):
            self._append_html(entry)
        self._maybe_autoscroll()

    def clear(self):
        self._entries.clear()
        self.view.clear()

    # ---- 渲染 ----
    def _passes(self, entry) -> bool:
        _, level = LOG_FILTERS[self.filter_combo.currentIndex()]
        return level is None or entry[1] == level

    @staticmethod
    def _entry_html(entry) -> str:
        ts, level, text, _tag = entry
        esc = _html.escape(text)
        color = LOG_LINE_COLOR.get(level, C["text"])
        return (f'<span style="color:#98A1B3;">{_html.escape(ts)}</span> '
                f'<span style="color:{color};">{esc}</span>')

    def _append_html(self, entry):
        # QTextEdit.append 对含 HTML 标签的文本按富文本插入（PySide6 无 appendHtml）
        self.view.append(self._entry_html(entry))

    def _render_all(self):
        self.view.clear()
        for entry in self._entries:
            if self._passes(entry):
                self._append_html(entry)
        self._maybe_autoscroll()

    # ---- 交互 ----
    def _on_scroll(self, value):
        sb = self.view.verticalScrollBar()
        at_bottom = value >= sb.maximum() - 2
        if self.autoscroll_chk.isChecked() and not at_bottom and sb.maximum() > 0:
            self.autoscroll_chk.blockSignals(True)
            self.autoscroll_chk.setChecked(False)
            self.autoscroll_chk.blockSignals(False)

    def _on_autoscroll_toggled(self, on):
        if on:
            self._maybe_autoscroll()

    def _maybe_autoscroll(self):
        if self.autoscroll_chk.isChecked():
            sb = self.view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _copy_selected(self):
        sel = self.view.textCursor().selectedText().replace("\u2029", "\n")
        if sel.strip():
            QApplication.clipboard().setText(sel)

    def _copy_all(self):
        QApplication.clipboard().setText("\n".join(t for _, _, t, _ in self._entries))

    def focus_log(self):
        self.view.setFocus()

    def reveal_file(self, name: str):
        """「查看原因」：切回全部级别并定位到该文件的日志段。"""
        idx = self.filter_combo.findText("全部")
        self.filter_combo.blockSignals(True)
        self.filter_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.filter_combo.blockSignals(False)
        self._render_all()
        target = None
        for i, (_ts, _lv, text, tag) in enumerate(self._entries):
            if tag == name or text.startswith(f"[失败] {name}"):
                target = i
                if text.startswith("[失败]"):
                    break
        if target is None:
            return
        doc = self.view.document()
        block = doc.findBlockByNumber(target)
        if block.isValid():
            cursor = self.view.textCursor()
            cursor.setPosition(block.position())
            self.view.setTextCursor(cursor)
            self.view.ensureCursorVisible()
            self.view.setFocus()


# ================= 状态 chip 辅助 =================
def chip_html(text: str, dot_color: str) -> str:
    """圆点 + 文本（状态不只靠颜色表达）。"""
    return (f'<span style="color:{dot_color};">●</span> '
            f'<span style="color:{C["text"]};">{_html.escape(text)}</span>')


# ================= 关闭对话框 =================
class CloseDialog(QDialog):
    CHOICE_MINIMIZE = 1
    CHOICE_QUIT = 2

    def __init__(self, parent, running: bool, tray_available: bool):
        super().__init__(parent)
        self.setWindowTitle("关闭 课堂笔记管家")
        self.setMinimumWidth(440)
        self.choice = None
        v = QVBoxLayout(self)
        v.setSpacing(12)
        msg = QLabel(self._message(running, tray_available))
        msg.setWordWrap(True)
        v.addWidget(msg)

        btn_min = QPushButton("最小化到托盘（继续监视）")
        btn_min.clicked.connect(lambda: self._done(self.CHOICE_MINIMIZE))
        btn_quit = QPushButton("退出应用")
        btn_quit.setObjectName("primaryBtn")
        btn_quit.clicked.connect(lambda: self._done(self.CHOICE_QUIT))
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)

        btns = QHBoxLayout()
        if tray_available:
            btns.addWidget(btn_min)
        btns.addStretch()
        btns.addWidget(btn_quit)
        btns.addWidget(btn_cancel)
        v.addLayout(btns)
        if tray_available:
            btn_min.setDefault(True)
        else:
            btn_cancel.setDefault(True)

    @staticmethod
    def _message(running: bool, tray_available: bool) -> str:
        lines = []
        if running:
            lines.append("正在处理任务。退出应用不会立即中断当前步骤，"
                         "会在当前步骤的安全结束点后退出；期间界面保持响应。")
            lines.append("")
        if tray_available:
            lines.append("最小化到托盘可继续监视收件箱。")
        else:
            lines.append("系统托盘不可用，无法最小化到托盘。只能退出应用或取消。")
        return "\n".join(lines)

    def _done(self, choice):
        self.choice = choice
        self.accept()


# ================= 设置对话框 =================
class SettingsDialog(QDialog):
    def __init__(self, parent=None, watch_running=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(520)
        self._watch_running = watch_running or (lambda: False)
        self._probe_thread = None

        form = QFormLayout(self)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)

        self.inbox_edit = QLineEdit(core.CFG["inbox_dir"])
        self.vault_edit = QLineEdit(core.CFG["vault_dir"])
        self.inbox_error = QLabel("")
        self.inbox_error.setObjectName("inlineError")
        self.vault_error = QLabel("")
        self.vault_error.setObjectName("inlineError")

        self.whisper_combo = QComboBox()
        items, cur_idx = whisper_combo_entries(
            core.CFG.get("models_dir", ""), core.CFG.get("whisper_model", ""))
        for label, value in items:
            self.whisper_combo.addItem(label, value)
        self.whisper_combo.setCurrentIndex(cur_idx)

        self.ollama_edit = QLineEdit(core.CFG["ollama_model"])
        self.engine_status = QLabel("AI 引擎：检测中…")
        self.engine_status.setObjectName("hint")

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

        def path_row(edit, btn):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            h.addWidget(edit, 1)
            h.addWidget(btn)
            return w

        form.addRow("收件箱目录（LocalSend 导入处）", path_row(self.inbox_edit, btn_inbox))
        form.addRow("", self.inbox_error)
        form.addRow("Obsidian vault 目录", path_row(self.vault_edit, btn_vault))
        form.addRow("", self.vault_error)
        form.addRow("转写模型", self.whisper_combo)
        form.addRow("纪要模型（Ollama）", self.ollama_edit)
        form.addRow("", self.engine_status)

        btns = QHBoxLayout()
        ok = QPushButton("保存")
        ok.setObjectName("primaryBtn")
        cancel = QPushButton("取消")
        ok.clicked.connect(self._on_save)
        cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        form.addRow(btns)

        self._probe_engine()

    # ---- 异步探测 Ollama 模型（不阻塞对话框） ----
    def _probe_engine(self):
        self._probe_thread = EngineProbe()
        self._probe_thread.probed.connect(self._on_probed)
        self._probe_thread.finished.connect(self._probe_thread.deleteLater)
        self._probe_thread.start()

    def _on_probed(self, state: str):
        model = self.ollama_edit.text().strip()
        if state in (ENGINE_READY, ENGINE_READY_CPU):
            if model and model != core.CFG.get("ollama_model", ""):
                self.engine_status.setText(f"AI 引擎就绪；注意：{model} 与当前配置不同，保存后需 Ollama 已拉取该模型")
            elif state == ENGINE_READY_CPU:
                self.engine_status.setText(
                    "AI 引擎就绪，但**跑在 CPU 上**（GPU 发现失败）：纪要会明显变慢、"
                    "上下文被压缩，质量可能下降。建议检查显卡驱动后重启本应用")
            else:
                self.engine_status.setText("AI 引擎就绪，模型可用")
        elif state == ENGINE_MODEL_MISSING:
            self.engine_status.setText(f"警告：Ollama 正在运行，但未找到模型 {model}，纪要阶段会失败")
        else:
            self.engine_status.setText("AI 引擎不可用（Ollama 未运行），纪要阶段会失败")

    # ---- 校验 + 保存 ----
    def _validate(self) -> bool:
        ok = True
        for edit, err, name in ((self.inbox_edit, self.inbox_error, "收件箱目录"),
                                (self.vault_edit, self.vault_error, "Obsidian vault 目录")):
            p = Path(edit.text().strip())
            if not edit.text().strip():
                edit.setProperty("invalid", True)
                err.setText(f"{name}不能为空")
                ok = False
            elif not p.is_dir():
                edit.setProperty("invalid", True)
                err.setText(f"{name}不存在：{edit.text().strip()}")
                ok = False
            else:
                edit.setProperty("invalid", False)
                err.setText("")
            edit.style().unpolish(edit)
            edit.style().polish(edit)
        return ok

    def _on_save(self):
        if not self._validate():
            return
        self.save()
        self.accept()

    def save(self):
        """落盘到唯一用户配置 core.CONFIG_PATH。
        监视运行中只落盘、不改运行中的 core.CFG（防止 inbox 仍旧、vault 已新的
        半生效状态），由「重启监视后生效」统一应用；监视未运行时同步内存配置。"""
        cfg = dict(core.CFG)
        cfg["inbox_dir"] = self.inbox_edit.text().strip()
        cfg["vault_dir"] = self.vault_edit.text().strip()
        cfg["whisper_model"] = self.whisper_combo.currentData()
        cfg["ollama_model"] = self.ollama_edit.text().strip()
        with open(core.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        if not self._watch_running():
            core.CFG.update(cfg)


# ================= 手动转录预览对话框 =================
class ManualPreviewDialog(QDialog):
    """展示待加入列表与重复 / 不支持格式提示；不支持的格式不复制。"""

    def __init__(self, parent, rows):
        super().__init__(parent)
        self.setWindowTitle("手动转录 · 确认加入")
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        v.setSpacing(8)
        hint = QLabel("以下文件将复制到收件箱并立即转录（会自动启动监视）：")
        hint.setWordWrap(True)
        v.addWidget(hint)

        table = QTableWidget(len(rows), 2)
        self.table = table
        table.setHorizontalHeaderLabels(["文件", "处理方式"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.verticalHeader().setDefaultSectionSize(ROW_H - 8)
        for i, (name, action, color) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(name))
            it = QTableWidgetItem(action)
            it.setForeground(QColor(color))
            table.setItem(i, 1, it)
        v.addWidget(table)

        btns = QHBoxLayout()
        ok = QPushButton("加入")
        ok.setObjectName("primaryBtn")
        cancel = QPushButton("取消")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)


# ================= 主窗口 =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(760, 500)
        self.setAcceptDrops(True)

        self.worker = None
        self._rows = {}           # 文件名 -> 行号
        self._status = {}         # 文件名 -> 最近状态文本
        self._note_dirs = {}      # 文件名 -> 笔记目录
        self._active = set()      # 进行中 / 排队的文件名
        self._batch = {"success": 0, "fail": 0, "skip": 0, "started": False}
        self._watch_state = WATCH_NOT_STARTED
        self._engine_state = ENGINE_STARTING
        self._engine_thread = None
        self._quitting = False
        self._quit_timer = None
        self._transient_timer = None
        self._statusbar_failure_latched = False
        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._note_dir = ""       # 当前处理文件的笔记目录

        self._build_ui()
        self._setup_tray()
        self._setup_shortcuts()
        self._restore_window()
        self._set_watch_state(WATCH_NOT_STARTED)
        self._set_engine_state(ENGINE_STARTING)
        self._set_statusbar_idle()
        # 启动探测：2 秒后开始，45 秒宽限期内持续重试（Ollama 冷启动 5s+），
        # 宽限期内显示「启动中」，超时才判「不可用」
        QTimer.singleShot(2000, lambda: self._probe_engine(45.0))

    # ---------- UI 构建 ----------
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # ==== 头部：产品名 + 收件箱路径 + AI 引擎状态 + 监视状态 ====
        head = QHBoxLayout()
        head.setSpacing(12)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        self.sub_label = QLabel()
        self.sub_label.setObjectName("appSub")
        title_box.addWidget(title)
        title_box.addWidget(self.sub_label)
        head.addLayout(title_box)
        head.addStretch()
        self.engine_chip = QLabel()
        self.engine_chip.setObjectName("chip")
        self.watch_chip = QLabel()
        self.watch_chip.setObjectName("chip")
        head.addWidget(self.engine_chip)
        head.addWidget(self.watch_chip)
        root.addLayout(head)

        # ==== 动作栏 ====
        bar_card = QFrame()
        bar_card.setObjectName("card")
        bar = QHBoxLayout(bar_card)
        bar.setContentsMargins(16, 10, 16, 10)
        bar.setSpacing(8)
        self.btn_toggle = QPushButton("开始监视")
        self.btn_toggle.setObjectName("primaryBtn")
        self.btn_toggle.setToolTip("开始 / 停止监视收件箱（F5）")
        self.btn_manual = QPushButton("手动转录")
        self.btn_manual.setToolTip("手动选择音频文件转录（会自动启动监视；Ctrl+O）")
        self.btn_inbox = QPushButton("打开收件箱")
        self.btn_inbox.setToolTip("在资源管理器中打开收件箱目录")
        self.btn_vault = QPushButton("打开 Obsidian")
        self.btn_vault.setToolTip("在 Obsidian / 资源管理器中打开笔记库")
        self.btn_settings = QPushButton("设置")
        self.btn_settings.setToolTip("收件箱 / 笔记库 / 转写与纪要模型（Ctrl+,）")
        self.btn_toggle.clicked.connect(self.toggle_watch)
        self.btn_manual.clicked.connect(self.open_manual_transcribe)
        self.btn_inbox.clicked.connect(self.open_inbox)
        self.btn_vault.clicked.connect(self.open_vault)
        self.btn_settings.clicked.connect(self.open_settings)
        for b in (self.btn_toggle, self.btn_manual, self.btn_inbox,
                  self.btn_vault, self.btn_settings):
            bar.addWidget(b)
        bar.addStretch()
        self.sub_label.setText(f"收件箱：{core.CFG['inbox_dir']}")
        root.addWidget(bar_card)

        # ==== 任务卡片 + 日志卡片（QSplitter，默认约 3:2） ====
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)

        task_card = QFrame()
        task_card.setObjectName("card")
        tv = QVBoxLayout(task_card)
        tv.setContentsMargins(16, 12, 16, 12)
        tv.setSpacing(8)
        task_head = QHBoxLayout()
        task_head.setSpacing(8)
        self.task_title = QLabel("任务（已完成 0 / 共 0）")
        self.task_title.setObjectName("cardTitle")
        task_head.addWidget(self.task_title)
        task_head.addStretch()
        btn_clear_tasks = QPushButton("清空记录")
        btn_clear_tasks.setProperty("flat", True)
        btn_clear_tasks.setToolTip("仅清空界面记录，不删除任何音频、笔记或日志文件")
        btn_clear_tasks.clicked.connect(self._clear_tasks)
        task_head.addWidget(btn_clear_tasks)
        tv.addLayout(task_head)

        self.task_stack = QStackedWidget()
        # 空状态页
        empty_page = QWidget()
        ev = QVBoxLayout(empty_page)
        ev.setAlignment(Qt.AlignCenter)
        ev.setSpacing(8)
        empty_title = QLabel("把录音放进收件箱即可自动开始")
        empty_title.setAlignment(Qt.AlignCenter)
        f = empty_title.font()
        f.setPointSize(11)
        f.setWeight(QFont.DemiBold)
        empty_title.setFont(f)
        self.empty_inbox = QLabel(core.CFG["inbox_dir"])
        self.empty_inbox.setObjectName("hint")
        self.empty_inbox.setAlignment(Qt.AlignCenter)
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignCenter)
        btn_row.setSpacing(8)
        ebtn_inbox = QPushButton("打开收件箱")
        ebtn_inbox.clicked.connect(self.open_inbox)
        ebtn_manual = QPushButton("手动选择音频")
        ebtn_manual.clicked.connect(self.open_manual_transcribe)
        btn_row.addWidget(ebtn_inbox)
        btn_row.addWidget(ebtn_manual)
        ev.addWidget(empty_title)
        ev.addWidget(self.empty_inbox)
        ev.addLayout(btn_row)
        # 表格页
        table_page = QWidget()
        tp = QVBoxLayout(table_page)
        tp.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["录音文件", "学科", "笔记标题", "状态", "耗时", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(3, 150)
        self.table.setColumnWidth(4, 70)
        self.table.setColumnWidth(5, 120)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        tp.addWidget(self.table)
        self.task_stack.addWidget(empty_page)
        self.task_stack.addWidget(table_page)
        tv.addWidget(self.task_stack)
        self.splitter.addWidget(task_card)

        self.log_view = LogView()
        log_card = QFrame()
        log_card.setObjectName("card")
        lv = QVBoxLayout(log_card)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.log_view)
        self.splitter.addWidget(log_card)

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([380, 250])
        root.addWidget(self.splitter, 1)

        # ==== 底部状态条 ====
        status_bar = QFrame()
        status_bar.setObjectName("statusBar")
        sb = QHBoxLayout(status_bar)
        sb.setContentsMargins(16, 10, 16, 10)
        sb.setSpacing(12)
        self.status_file = QLabel("当前无任务")
        self.status_file.setMinimumWidth(220)
        self.status_phase = QLabel("")
        self.status_phase.setObjectName("hint")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(260)
        self.status_pct = QLabel("")
        self.status_pct.setObjectName("hint")
        self.status_pct.setMinimumWidth(64)
        sb.addWidget(self.status_file)
        sb.addWidget(self.status_phase)
        sb.addStretch()
        sb.addWidget(self.progress)
        sb.addWidget(self.status_pct)
        root.addWidget(status_bar)

        self.setCentralWidget(central)

    # ---------- 托盘 ----------
    def _setup_tray(self):
        self.tray = None
        self._tray_available = QSystemTrayIcon.isSystemTrayAvailable()
        if not self._tray_available:
            return
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.windowIcon())
        menu = QMenu()
        act_show = QAction("显示主窗口", self)
        act_show.triggered.connect(self.show_normal)
        self.tray_toggle = QAction("开始监视", self)
        self.tray_toggle.triggered.connect(self.toggle_watch)
        act_inbox = QAction("打开收件箱", self)
        act_inbox.triggered.connect(self.open_inbox)
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self.quit_app)
        menu.addAction(act_show)
        menu.addAction(self.tray_toggle)
        menu.addAction(act_inbox)
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self.show_normal()
            if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick) else None
        )
        self.tray.show()
        self._update_tray_tooltip()

    def _update_tray_tooltip(self):
        if not self.tray:
            return
        n = len(self._active)
        tip = f"{APP_NAME} · {self._watch_state}"
        if n:
            tip += f"\n任务进行中：{n}"
        self.tray.setToolTip(tip)

    # ---------- 快捷键 ----------
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("F5"), self, activated=self.toggle_watch)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_manual_transcribe)
        QShortcut(QKeySequence("Ctrl+,"), self, activated=self.open_settings)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.log_view.focus_log)

    # ---------- 窗口几何 ----------
    def _restore_window(self):
        geo = self._settings.value("window/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(920, 620)
        sp = self._settings.value("window/splitter")
        if sp is not None:
            self.splitter.restoreState(sp)

    def _save_window(self):
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/splitter", self.splitter.saveState())

    def show_normal(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    # ---------- 引擎 / 状态 chip ----------
    def _probe_engine(self, grace_seconds: float = 0.0):
        if self._engine_thread is not None and self._engine_thread.isRunning():
            return   # 已有探测在进行，避免叠加
        self._engine_thread = EngineProbe(grace_seconds=grace_seconds)
        self._engine_thread.probed.connect(self._on_engine_probed)
        self._engine_thread.finished.connect(self._on_engine_thread_done)
        self._engine_thread.finished.connect(self._engine_thread.deleteLater)
        self._engine_thread.start()

    def _on_engine_probed(self, state: str):
        self._set_engine_state(state)
        if state == ENGINE_UNAVAILABLE and not self._quitting:
            # 不可用不是终局：之后每 60 秒静默复测一次（用户可能手动开了
            # Ollama，或首次拉起失败后重开应用）
            QTimer.singleShot(60000, lambda: self._probe_engine(0.0))
        # 注意：绝不能在这里 self._engine_thread = None！
        # probed 是排队信号，主线程执行本槽时 run() 可能刚返回、线程还在收尾，
        # 置 None 会丢掉 QThread 的唯一 Python 引用 → C++ 对象被立即析构 →
        # "QThread: Destroyed while thread is still running" → Qt 6.11 fail-fast 闪退。
        # 引用保留到 finished（线程彻底结束）后由 _on_engine_thread_done 清理。

    def _on_engine_thread_done(self):
        self._engine_thread = None

    def _set_engine_state(self, state: str):
        self._engine_state = state
        color = {
            ENGINE_READY: C["success"],
            ENGINE_READY_CPU: C["warning"],
            ENGINE_STARTING: C["brand"],
            ENGINE_MODEL_MISSING: C["warning"],
            ENGINE_UNAVAILABLE: C["danger"],
        }.get(state, C["text_secondary"])
        self.engine_chip.setText(chip_html(f"AI 引擎 · {state}", color))

    def _set_watch_state(self, state: str):
        self._watch_state = state
        color = {
            WATCH_WATCHING: C["success"],
            WATCH_STOPPING: C["warning"],
            WATCH_NOT_STARTED: C["text_secondary"],
            WATCH_STOPPED: C["text_secondary"],
        }.get(state, C["text_secondary"])
        self.watch_chip.setText(chip_html(state, color))
        self.btn_toggle.setEnabled(state != WATCH_STOPPING)
        self.btn_toggle.setText("停止监视" if state in (WATCH_WATCHING, WATCH_STOPPING)
                                else "开始监视")
        if self.tray:
            self.tray_toggle.setText("停止监视" if state == WATCH_WATCHING else "开始监视")
        self._update_tray_tooltip()

    # ---------- 瞬时反馈（3 秒回落） ----------
    def _transient(self, text: str):
        self.status_file.setText(text)
        if self._transient_timer:
            self._transient_timer.stop()
        else:
            self._transient_timer = QTimer(self)
            self._transient_timer.setSingleShot(True)
            self._transient_timer.timeout.connect(self._transient_timeout)
        self._transient_timer.start(3000)

    def _transient_timeout(self):
        self._refresh_status_file_text()

    def _refresh_status_file_text(self):
        if self._active:
            # 显示正在处理或最早排队的文件
            name = sorted(self._active)[0]
            self.status_file.setText(f"当前文件：{name}")
        else:
            self.status_file.setText("当前无任务")

    def _set_statusbar_idle(self):
        if self._statusbar_failure_latched:
            return
        self._refresh_status_file_text()
        self.status_phase.setText("")
        self.status_pct.setText("")

    # ---------- 监视控制 ----------
    def toggle_watch(self):
        if self._watch_state == WATCH_WATCHING:
            self.stop_watch()
        elif self._watch_state == WATCH_STOPPING:
            pass
        else:
            self.start_watch()

    def start_watch(self):
        if self._quitting:
            return
        if self.worker and self.worker.isRunning():
            return
        if self._engine_state in (ENGINE_MODEL_MISSING, ENGINE_UNAVAILABLE):
            r = QMessageBox.question(
                self, APP_NAME,
                f"AI 引擎当前状态：{self._engine_state}。\n"
                "继续监视的话，录音仍会转写归档，但 AI 纪要阶段会失败。\n"
                "是否仍要开始监视？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        core.STOP_FLAG = False
        self.worker = Worker()
        self.worker.log_line.connect(self.log_view.append_line)
        self.worker.task_state.connect(self.on_task_state)
        self.worker.task_meta.connect(self.on_task_meta)
        self.worker.task_done.connect(self.on_task_done)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        self._set_watch_state(next_watch_state(self._watch_state, "start"))
        self._probe_engine()

    def stop_watch(self):
        """异步停止：不阻塞 UI，worker 处理完当前文件后自动退出。"""
        if self.worker and self.worker.isRunning():
            core.request_stop()
            self._set_watch_state(next_watch_state(self._watch_state, "stopping"))
            self.log_view.append_line("[监视] 已请求停止，等待当前步骤安全结束…")

    def _on_worker_finished(self):
        self._set_watch_state(next_watch_state(self._watch_state, "stopped"))
        self._set_statusbar_idle()
        self.worker = None

    # ---------- 设置 ----------
    def open_settings(self):
        running = bool(self.worker and self.worker.isRunning())
        dlg = SettingsDialog(self, watch_running=lambda: running)
        if dlg.exec():
            # SettingsDialog._on_save 已完成唯一一次校验与落盘；这里仅刷新界面。
            self.sub_label.setText(f"收件箱：{core.CFG['inbox_dir']}")
            self.empty_inbox.setText(core.CFG["inbox_dir"])
            # 瞬时反馈；持续状态不变（next_watch_state 的 save 事件不改变状态）
            self._set_watch_state(next_watch_state(self._watch_state, "save"))
            self._transient("已保存 · 重启监视后生效")

    # ---------- 手动转录 / 拖放 ----------
    def open_manual_transcribe(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择要转录的音频（可多选）", "",
            "音频文件 (*.m4a *.mp3 *.wav *.aac *.flac *.opus *.ogg *.amr *.wma *.mp4)")
        if files:
            self._add_files([Path(f) for f in files])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()
                 if url.isLocalFile()]
        files = [p for p in paths if p.is_file()]   # 不接受文件夹
        if files:
            self._add_files(files)

    def _add_files(self, paths):
        """多文件先展示待加入列表与重复 / 不支持提示；不支持格式不复制。"""
        if not paths:
            return
        inbox = Path(core.CFG["inbox_dir"])
        supported = {e.lower() for e in core.CFG.get("supported_ext", [])}
        rows, plan = [], []
        reserved_destinations = set()
        for p in paths:
            ext = p.suffix.lower()
            if ext not in supported:
                rows.append((p.name, "不支持 · 跳过", C["warning"]))
                continue
            # 同一批次内也要预留目标名，不能只检查 inbox 中原有文件。
            dst = unique_dst_name(inbox, p.name, reserved_destinations)
            reserved_destinations.add(str(dst).casefold())
            action = "加入" if dst == inbox / p.name else f"重命名为 {dst.name}"
            rows.append((p.name, action, C["text_secondary"] if "重命名" in action else C["success"]))
            plan.append((p, dst))
        dlg = ManualPreviewDialog(self, rows)
        if dlg.exec() != QDialog.Accepted or not plan:
            return
        if not (self.worker and self.worker.isRunning()):
            self.start_watch()
            if not (self.worker and self.worker.isRunning()):
                self._transient("监视未能启动，请稍候再试")
                return
        copied = []
        for src, dst in plan:
            try:
                shutil.copy2(str(src), str(dst))
            except Exception as e:
                self.log_view.append_line(f"[失败] {src.name}: 复制到收件箱失败（{e}）")
                continue
            copied.append(dst.name)
        for n in copied:
            self.worker.enqueue(n)
            self._ensure_row(n)
            self._set_row_status(n, "排队中")
            self._active.add(n)
            self._note_dirs.pop(n, None)
        if copied:
            self._batch_start_if_needed()
            self.log_view.append_line(f"[手动] 加入转录：{', '.join(copied)}")
            self._transient(f"已加入 {len(copied)} 个文件")
        self._update_tray_tooltip()

    def open_inbox(self):
        self._open_dir(core.CFG["inbox_dir"])

    def open_vault(self):
        """打开笔记库。
        优先让已在运行的 Obsidian 聚焦并打开该 vault；Obsidian 没装/没开再退回资源管理器。
        （2026-09-17 修复：此前统一走 QDesktopServices.openUrl(QUrl.fromLocalFile(文件夹))，
        在 Windows 上对**目录**的关联行为不稳定，实测会落到别的 shell 处理程序上去，
        用户看到的是"点开 Obsidian 却弹出了任务管理器/别的窗口"。）
        """
        vault = Path(core.CFG["vault_dir"])
        if not vault.is_dir():
            QMessageBox.warning(self, APP_NAME, f"目录不存在：{vault}")
            return
        if self._launch_obsidian(vault):
            self._transient("已在 Obsidian 中打开笔记库")
            return
        # 退路：资源管理器打开该目录（用 ShellExecute 而非 QDesktopServices，见 _shell_open）
        if self._shell_open(vault):
            self._transient("未找到 Obsidian，已在资源管理器中打开笔记库")
        else:
            QMessageBox.warning(
                self, APP_NAME,
                f"无法打开笔记库目录：\n{vault}\n\n"
                "请手动在资源管理器中打开该路径。")

    # ---------- 外部程序调用 ----------
    @staticmethod
    def _obsidian_exe():
        """定位 Obsidian.exe（注册表 URL 协议 → 常见安装路径）。返回 Path 或 None。"""
        # ① 从 obsidian:// 协议注册读取真实 exe 路径（最可靠，跟安装位置无关）
        try:
            import winreg
            for root, sub in (
                (winreg.HKEY_CLASSES_ROOT, r"obsidian\shell\open\command"),
                (winreg.HKEY_CURRENT_USER, r"Software\Classes\obsidian\shell\open\command"),
            ):
                try:
                    with winreg.OpenKey(root, sub) as k:
                        cmd = winreg.QueryValueEx(k, "")[0]
                except OSError:
                    continue
                m = re.search(r'"([^"]+\.exe)"', cmd) or re.search(r"(\S+\.exe)", cmd)
                if m:
                    p = Path(m.group(1))
                    if p.is_file():
                        return p
        except Exception:
            pass
        # ② 常见安装位置兜底
        cands = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Obsidian" / "Obsidian.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "Obsidian" / "Obsidian.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Obsidian" / "Obsidian.exe",
        ]
        for p in cands:
            if p and p.is_file():
                return p
        return None

    def _launch_obsidian(self, vault: Path) -> bool:
        """用 obsidian:// 深链打开 vault。成功返回 True。
        深链由已运行的 Obsidian 单实例接管（不会新开窗口）；Obsidian 没运行时
        协议处理器会自行拉起它。进程用 DETACHED_PROCESS 启动，关闭本程序不会连带杀掉 Obsidian。
        """
        exe = self._obsidian_exe()
        if exe is None:
            return False
        uri = "obsidian://open?path=" + quote(str(vault))
        try:
            DETACHED = 0x00000008
            subprocess.Popen(
                [str(exe), uri],
                close_fds=True,
                creationflags=DETACHED | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            return True
        except Exception as e:
            self.log_view.append_line(f"[打开] Obsidian 启动失败：{e}")
            return False

    @staticmethod
    def _shell_open(p: Path) -> bool:
        """用 ShellExecuteW 打开文件/目录（比 QDesktopServices 在 Windows 上更贴近系统默认行为）。
        ShellExecuteW 对目录固定走资源管理器；对文件走用户设定的默认程序。"""
        try:
            import ctypes
            op = ctypes.windll.shell32.ShellExecuteW
            op.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                           ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int]
            op.restype = ctypes.c_void_p          # 返回 HINSTANCE，必须按指针取，否则 64 位下截断
            r = op(None, "open", str(p), None, None, 1)
            return int(r or 0) > 32               # ShellExecute 约定：>32 才是成功
        except Exception:
            return False

    def _open_dir(self, path):
        p = Path(path)
        if not p.is_dir():
            QMessageBox.warning(self, APP_NAME, f"目录不存在：{path}")
            return
        if not self._shell_open(p):
            # 最后退路：Qt 的方式（Linux/macOS 上主要靠它）
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    # ---------- 任务表 ----------
    def _ensure_row(self, name) -> int:
        if name in self._rows:
            return self._rows[name]
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._rows[name] = row
        self.table.setItem(row, 0, QTableWidgetItem(name))
        self.table.setItem(row, 1, QTableWidgetItem(""))
        self.table.setItem(row, 2, QTableWidgetItem(""))
        self.table.setItem(row, 4, QTableWidgetItem(""))
        self._set_row_status(name, "排队中")
        self._active.add(name)
        self._batch_start_if_needed()
        self.task_stack.setCurrentIndex(1)
        self._update_counts()
        self._update_tray_tooltip()
        return row

    STATUS_DOT = {
        "排队中": C["text_secondary"],
        "完成": C["success"],
        "失败": C["danger"],
        "已跳过（重复）": C["warning"],
        "已取消": C["text_secondary"],
    }

    def _status_dot_color(self, text: str) -> str:
        if text in self.STATUS_DOT:
            return self.STATUS_DOT[text]
        if text.startswith(("转写中", "AI 整理", "归档", "知识补全")):
            return C["brand"]
        return C["text_secondary"]

    def _set_row_status(self, name, text):
        row = self._rows.get(name)
        if row is None:
            return
        self._status[name] = text
        lbl = QLabel(chip_html(text, self._status_dot_color(text)))
        lbl.setStyleSheet("background: transparent;")
        self.table.setCellWidget(row, 3, lbl)

    def on_task_state(self, name, state):
        self._ensure_row(name)
        self._set_row_status(name, state)

    def on_task_meta(self, name, subject, title):
        row = self._ensure_row(name)
        if subject:
            self.table.setItem(row, 1, QTableWidgetItem(subject))
        if title:
            self.table.setItem(row, 2, QTableWidgetItem(title))

    def on_progress(self, name, phase, pct):
        self._ensure_row(name)
        self._statusbar_failure_latched = False
        self.progress.setProperty("danger", False)
        self.progress.style().unpolish(self.progress)
        self.progress.style().polish(self.progress)
        self.status_file.setText(f"当前文件：{name}")
        self.status_phase.setText(phase)
        if pct < 0:
            self.progress.setRange(0, 0)   # 不确定进度（AI 整理 / 知识补全等）
            self.status_pct.setText("不可预估")
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(pct)
            self.status_pct.setText(f"{pct}%")

    def on_task_done(self, name, status, seconds, note_dir):
        self._ensure_row(name)
        self._set_row_status(name, status)
        self._note_dirs[name] = note_dir
        row = self._rows[name]
        if status != "失败":
            self.progress.setProperty("danger", False)
            self.progress.style().unpolish(self.progress)
            self.progress.style().polish(self.progress)
        if status == "完成":
            self.table.setItem(row, 4, QTableWidgetItem(f"{seconds:.0f}s"))
            self._add_row_button(name, "打开目录", lambda _=False, d=note_dir: self._open_dir(d))
        elif status == "失败":
            self.table.setItem(row, 4, QTableWidgetItem(f"{seconds:.0f}s"))
            self._add_row_button(name, "查看原因", lambda _=False, n=name: self.log_view.reveal_file(n))
            self._mark_progress_danger()
            self.status_file.setText(f"当前文件：{name}")
            self._statusbar_failure_latched = True
        self._active.discard(name)
        self._count_batch(status)
        self._update_counts()
        self._update_tray_tooltip()
        if not self._active:
            # 失败状态需要留在底部状态条，直到下一项任务或用户主动清空；
            # 否则红色进度刚显示就会被空闲状态覆盖。
            if status != "失败":
                self._set_statusbar_idle()
            self._notify_batch_done()

    def _mark_progress_danger(self):
        """失败：进度变红并停留在失败前的进度，不归零。"""
        self.progress.setProperty("danger", True)
        self.progress.style().unpolish(self.progress)
        self.progress.style().polish(self.progress)
        self.status_phase.setText("失败")
        if not self.status_pct.text():
            self.status_pct.setText("")

    def _add_row_button(self, name, text, handler):
        row = self._rows[name]
        btn = QPushButton(text)
        btn.setProperty("flat", True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(handler)
        self.table.setCellWidget(row, 5, btn)

    def _update_counts(self):
        terminal = ("完成", "失败", "已跳过（重复）", "已取消")
        done = sum(1 for s in self._status.values() if s in terminal)
        total = len(self._status)
        self.task_title.setText(f"任务（已完成 {done} / 共 {total}）")

    def _clear_tasks(self):
        """仅清空 UI 记录，不删除任何音频、笔记或日志文件。"""
        self.table.setRowCount(0)
        self._rows.clear()
        self._status.clear()
        self._note_dirs.clear()
        self._statusbar_failure_latched = False
        self.progress.setProperty("danger", False)
        self.progress.style().unpolish(self.progress)
        self.progress.style().polish(self.progress)
        self.task_stack.setCurrentIndex(0)
        self._update_counts()
        self._refresh_status_file_text()

    # ---------- 批次通知 ----------
    def _batch_start_if_needed(self):
        if not self._batch["started"]:
            self._batch = {"success": 0, "fail": 0, "skip": 0, "started": True}

    def _count_batch(self, status):
        if status == "完成":
            self._batch["success"] += 1
        elif status == "失败":
            self._batch["fail"] += 1
        elif status in ("已跳过（重复）", "已取消"):
            self._batch["skip"] += 1

    def _notify_batch_done(self):
        b = self._batch
        b["started"] = False
        if b["success"] or b["fail"] or b["skip"]:
            if self.tray:
                level = QSystemTrayIcon.Information if not b["fail"] else QSystemTrayIcon.Warning
                self.tray.showMessage(
                    APP_NAME,
                    f"本批任务全部完成：成功 {b['success']}，失败 {b['fail']}，跳过 {b['skip']}",
                    level, 4000)

    # ---------- 关闭 / 退出 ----------
    def closeEvent(self, event):
        if self._quitting:
            event.accept()
            return
        self._save_window()
        running = bool(self.worker and self.worker.isRunning())
        dlg = CloseDialog(self, running, self._tray_available)
        if dlg.exec() != QDialog.Accepted or dlg.choice is None:
            event.ignore()
            return
        if dlg.choice == CloseDialog.CHOICE_MINIMIZE:
            event.ignore()
            self.hide()
        else:
            event.ignore()
            self.quit_app()

    def quit_app(self):
        """事件驱动的安全退出：不阻塞主线程，等待 worker 到安全点。"""
        if self._quitting:
            return
        if self.worker and self.worker.isRunning():
            self._quitting = True
            core.request_stop()
            self._transient("正在等待当前步骤结束…")
            self.btn_toggle.setEnabled(False)
            self.worker.finished.connect(self._finish_quit)
            self._quit_timer = QTimer(self)
            self._quit_timer.setSingleShot(True)
            self._quit_timer.timeout.connect(self._ask_force_quit)
            self._quit_timer.start(30000)
        else:
            self._finish_quit()

    def _finish_quit(self):
        self._quitting = True
        # 引擎探测线程优雅收尾（QThread 运行中析构 = Qt 6.11 fail-fast 闪退）
        t = getattr(self, "_engine_thread", None)
        if t is not None and t.isRunning():
            t.stop()
            t.wait(3000)
        self._save_window()
        if self.tray:
            self.tray.hide()
        QApplication.quit()

    def _ask_force_quit(self):
        if not self._quitting or not (self.worker and self.worker.isRunning()):
            return
        # 2026-09-17 修复：旧实现只有一个「警告 + 再等 30 秒」按钮，用户如果执意
        # 退出就陷入无限弹窗循环（每 30 秒弹一次，永远退不掉），只能去任务管理器杀进程。
        # 线程内部在 transcribe / generate_note / knowledge_patch 期间并不检查 STOP_FLAG，
        # 所以「等它自己结束」可能等非常久。这里改成一次性给用户明确选择：
        #   继续等待（默认，安全） / 立即强制退出（明确告知风险）。
        # 强制退出不再调 QThread.terminate（会打断文件写入造成半归档），
        # 而是直接结束进程——文件层面靠 archive() 的原子写入保证不产生半成品。
        self.log_view.append_line("[退出] 当前步骤超过 30 秒仍未结束，等待用户决定")
        box = QMessageBox(QMessageBox.Warning, APP_NAME,
                          "当前步骤仍未结束（可能正在转写或 AI 整理）。\n\n"
                          "· 继续等待：保护正在处理的文件完整性，但可能需要较长时间。\n"
                          "· 立即退出：程序会直接结束，当前正在处理的文件不会归档，"
                          "下次可重新放入收件箱处理（已归档的文件不受影响）。",
                          QMessageBox.NoButton, self)
        wait_btn = box.addButton("继续等待", QMessageBox.AcceptRole)
        exit_btn = box.addButton("立即退出", QMessageBox.DestructiveRole)
        box.setDefaultButton(wait_btn)
        box.exec()
        if box.clickedButton() is exit_btn:
            self.log_view.append_line("[退出] 用户选择立即退出，正在结束进程")
            self._force_quit()
        elif self._quitting and self.worker and self.worker.isRunning():
            self._quit_timer.start(30000)

    def _force_quit(self):
        """绕过 worker 的立即退出路径：先把进程内已归档的文件落盘（archive 已做原子写），
        再强杀自身。用 os._exit 而非 QApplication.quit —— 后者会被仍运行的 worker 线程拖住。"""
        try:
            self._save_window()
        except Exception:
            pass
        try:
            if self.tray:
                self.tray.hide()
        except Exception:
            pass
        # 若引擎探测线程还在，给它极短时间收尾；它不持有关键文件句柄
        t = getattr(self, "_engine_thread", None)
        if t is not None and t.isRunning():
            t.stop()
            t.wait(1500)
        os._exit(0)


# ================= 单实例（QLocalServer / QLocalSocket） =================
def _activate_existing_instance() -> bool:
    """已有一实例在运行 → 通知它显示窗口，返回 True（本次启动退出）。"""
    sock = QLocalSocket()
    sock.connectToServer(INSTANCE_KEY)
    if sock.waitForConnected(500):
        sock.write(b"show\n")
        sock.flush()
        sock.waitForBytesWritten(500)
        sock.disconnectFromServer()
        return True
    return False


def _app_icon():
    """窗口 / 任务栏 / 托盘图标。

    **必须显式设置**：Qt 在 Windows 上**不会**自动回落到 exe 内嵌图标。
    不设的话窗口标题栏和任务栏显示的是 Qt 自带默认图标（实测是个「文档」图），
    与 exe 图标不一致——用户看到的效果就是"图标根本没换"。
    另外 QIcon 也读不了 exe 内嵌图标（实测 QIcon(sys.executable) 返回空，
    QFileIconProvider 走 shell 才读得到），所以必须把 icon.ico 当数据文件
    一起打包，运行时按路径加载。
    """
    cands = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:                                              # 打包后：_internal/icon.ico
        cands.append(Path(meipass) / "icon.ico")
    if getattr(sys, "frozen", False):
        cands.append(Path(sys.executable).parent / "icon.ico")
    cands.append(Path(__file__).parent.parent / "icon.ico")  # 开发模式：项目根
    for p in cands:
        if p.exists():
            ic = QIcon(str(p))
            if not ic.isNull():
                print(f"[图标] 使用 {p}")
                return ic
    print("[图标] 未找到可用的 icon.ico，回落到 Qt 默认图标")
    return QIcon()


def main():
    # ---- 启动诊断日志（2026-09-17 长期保留）：stdout/stderr/Qt原生消息全落盘 ----
    # windowed exe 的 stdout/stderr 句柄无效，dup2 重定向到 tmp/app_boot.log 后，
    # Python traceback 与 Qt 告警/fatal 消息都能留档，闪退才有取证现场。
    try:
        import faulthandler
        import traceback as _tb
        if getattr(sys, "frozen", False):
            _root = Path(sys.executable).parent.parent.parent
        else:
            _root = Path(APP_DIR).parent
        (_root / "tmp").mkdir(parents=True, exist_ok=True)
        # 文本模式！二进制模式会让 print(str) 直接 TypeError（上一版诊断自坑）
        _bootlog = open(_root / "tmp" / "app_boot.log", "a", encoding="utf-8",
                        errors="replace", buffering=1)
        os.dup2(_bootlog.fileno(), 1)
        os.dup2(_bootlog.fileno(), 2)
        sys.stdout = _bootlog
        sys.stderr = _bootlog
        faulthandler.enable(file=_bootlog)
        print(f"\n===== BOOT {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} "
              f"frozen={getattr(sys, 'frozen', False)} =====", flush=True)
    except Exception as _e:
        print(f"[BOOT] 日志初始化失败: {_e}")

    print("[BOOT] 创建 QApplication ...", flush=True)
    app = QApplication(sys.argv)
    print("[BOOT] QApplication OK", flush=True)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(SETTINGS_ORG)
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(_app_icon())      # 窗口/任务栏/托盘图标（Qt 不会自动取 exe 图标）
    app.setStyleSheet(APP_QSS)
    app.setFont(QFont("Microsoft YaHei", 10))
    print("[BOOT] 图标/QSS/字体 OK", flush=True)

    # 单实例（2026-09-17 原子化修复）：QLocalServer 在 Windows 上同名可多监听，
    # 双开竞态会让两个实例都"单实例 OK"并同时轮询 inbox（双转写抢 GPU）。
    # 改用 Windows 命名互斥体做原子裁决：抢不到锁 = 已有活实例 → 唤醒后退出。
    import ctypes
    # 使用 use_last_error=True + 正确的 HANDLE 返回类型，避免 ctypes 默认 c_int
    # 截断 Windows 句柄；ERROR_ALREADY_EXISTS=183 是 CreateMutexW 的唯一判据。
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _create_mutex = _kernel32.CreateMutexW
    _create_mutex.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    _create_mutex.restype = ctypes.c_void_p
    _mutex = _create_mutex(None, False, "ClassNoteKeeper.SingleInstanceMutex")
    if not _mutex:
        raise ctypes.WinError(ctypes.get_last_error())
    _ERROR_ALREADY_EXISTS = 183
    _first = ctypes.get_last_error() != _ERROR_ALREADY_EXISTS
    if not _first:
        if _activate_existing_instance():
            sys.exit(0)
        # 抢锁失败且唤醒失败：互斥体存在说明有活实例（进程死亡会自动释放锁），安全退出
        print("[BOOT] 已有实例在运行但唤醒失败，退出", flush=True)
        sys.exit(0)
    QLocalServer.removeServer(INSTANCE_KEY)   # 清理可能残留的僵死命名管道
    instance_server = QLocalServer()
    if not instance_server.listen(INSTANCE_KEY):
        print(f"[BOOT] 单实例通信端口创建失败：{instance_server.errorString()}", flush=True)
        # 互斥体已经完成原子裁决；通信端口失败时仍继续启动，避免误把唯一实例判为启动失败。
    else:
        print("[BOOT] 单实例 OK", flush=True)

    # Ollama 后台拉起（不阻塞窗口显示，加快启动）
    ollama_mgr = OllamaManager()
    threading.Thread(target=ollama_mgr.ensure, daemon=True).start()
    app.aboutToQuit.connect(ollama_mgr.shutdown)

    win = MainWindow()
    print("[BOOT] MainWindow 构建完成", flush=True)
    win.show()
    print("[BOOT] 窗口已显示，进入事件循环", flush=True)

    def _on_new_connection():
        while instance_server.hasPendingConnections():
            conn = instance_server.nextPendingConnection()
            if not conn:
                continue

            def _wake(c=conn):
                try:
                    c.readAll()
                except Exception:
                    pass
                win.show_normal()
                c.disconnectFromServer()
                c.deleteLater()

            conn.readyRead.connect(_wake)
            conn.disconnected.connect(_wake)

    instance_server.newConnection.connect(_on_new_connection)

    print("[BOOT] 事件循环开始 exec", flush=True)
    _rc = app.exec()
    print(f"[BOOT] 事件循环退出 rc={_rc}", flush=True)
    sys.exit(_rc)


if __name__ == "__main__":
    main()
