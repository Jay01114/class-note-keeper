# -*- coding: utf-8 -*-
"""_wait_gpu_ready 回归测试（2026-09-17 GPU discovery 竞态修复）。

覆盖：
  ① 本次启动写入 library=CUDA  -> 立即 True
  ② 本次启动写入 library=cpu   -> False（不再等）
  ③ 日志里只有历史 CUDA 行（start_offset 之后为空）-> 不误判，超时 False
  ④ 日志里历史有 cpu 行、本次有 CUDA 行 -> True（关键：不命中历史）
  ⑤ 文件不存在                 -> False
  ⑥ 延迟写入 CUDA（模拟 20 秒发现）-> 轮询后 True
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import app as A  # noqa: E402

CUDA = ('time=2026-09-17T22:33:56.003+08:00 level=INFO source=types.go:32 '
        'msg="inference compute" id=0 filter_id=0 library=CUDA compute=12.0 '
        'name=CUDA0 description="NVIDIA GeForce RTX 5060 Laptop GPU"\n')
CPU = ('time=2026-09-17T22:28:43.891+08:00 level=INFO source=types.go:50 '
       'msg="inference compute" id=cpu library=cpu compute="" name=cpu '
       'total="31.3 GiB"\n')

results = []


def check(name, got, want):
    ok = got == want
    results.append((name, ok))
    print(f"  {'✓' if ok else '✗'} {name}  got={got} want={want}")


def T(name, fn):
    print(f"\n--- {name} ---")
    fn()


def t1():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.log"
        p.write_text("old\n", encoding="utf-8")
        off = p.stat().st_size
        p.write_text("old\n" + CUDA, encoding="utf-8")
        check("本次 CUDA -> True", A._wait_gpu_ready(p, off, timeout=3), True)


def t2():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.log"
        p.write_text("old\n", encoding="utf-8")
        off = p.stat().st_size
        p.write_text("old\n" + CPU, encoding="utf-8")
        t0 = time.time()
        r = A._wait_gpu_ready(p, off, timeout=10)
        dt = time.time() - t0
        check("本次 cpu -> False", r, False)
        check("cpu 分支立即返回（<2s）", dt < 2.0, True)


def t3():
    """历史有 CUDA，start_offset 之后无新内容 -> 不应误判为 True。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.log"
        p.write_text(CUDA, encoding="utf-8")
        off = p.stat().st_size          # 历史 CUDA 全部在 offset 之前
        check("纯历史 CUDA -> 超时 False", A._wait_gpu_ready(p, off, timeout=3), False)


def t4():
    """历史有 cpu、本次有 CUDA -> 必须 True（不命中历史 cpu 行）。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.log"
        p.write_text(CPU + CPU, encoding="utf-8")   # 上一轮失败的 CPU 行
        off = p.stat().st_size
        p.write_text(CPU + CPU + CUDA, encoding="utf-8")
        check("历史 cpu + 本次 CUDA -> True", A._wait_gpu_ready(p, off, timeout=5), True)


def t5():
    check("文件不存在 -> False",
          A._wait_gpu_ready(Path(r"D:\__no_such_dir__\nope.log"), 0, timeout=2), False)


def t6():
    """延迟 4 秒才写入 CUDA，模拟真实 GPU discovery 耗时。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.log"
        p.write_text("boot\n", encoding="utf-8")
        off = p.stat().st_size

        def later():
            time.sleep(4)
            with open(p, "a", encoding="utf-8") as f:
                f.write(CUDA)

        threading.Thread(target=later, daemon=True).start()
        t0 = time.time()
        r = A._wait_gpu_ready(p, off, timeout=15)
        dt = time.time() - t0
        check("延迟写入 -> True", r, True)
        check("等待约 4s（3s<dt<8s）", 3.0 < dt < 8.0, True)


T("① 本次 CUDA", t1)
T("② 本次 CPU", t2)
T("③ 纯历史 CUDA 不算数", t3)
T("④ 历史 CPU + 本次 CUDA", t4)
T("⑤ 文件缺失", t5)
T("⑥ 延迟写入", t6)

ok = sum(1 for _, c in results if c)
print(f"\n=== {ok}/{len(results)} 通过 ===")
if ok != len(results):
    print("失败：" + "、".join(n for n, c in results if not c))
    sys.exit(1)
