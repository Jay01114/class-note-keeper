# 从零搭建指南

本指南假设：Windows 11 + NVIDIA 显卡（有 CUDA）。其他系统（Linux/macOS）步骤类似，模型与依赖相同。

## 0. 前置条件

| 组件 | 要求 |
|---|---|
| 操作系统 | Windows 10/11（或 Linux/macOS） |
| Python | 3.11+（建议 3.11/3.12） |
| NVIDIA 显卡 | 建议 ≥ 8GB 显存（6GB 可用 small/medium 档） |
| 手机 | 任何能装 LocalSend 的手机（Android/iOS/鸿蒙） |

## 1. 安装 Ollama（纪要模型引擎）

从 Ollama 官方渠道安装（Windows 安装包 / 或 `curl -fsSL https://ollama.com/install.sh | sh`）：

```bash
# 拉取纪要模型（二选一）
ollama pull qwen2.5:7b    # 推荐，质量好，显存约 4.7GB
ollama pull qwen2.5:3b    # 轻量，更快，质量略降
```

> 国内网络可配置镜像：`export OLLAMA_HOST=...` 或使用 `ollama.com` 官方源；Qwen 模型本身在 `registry.ollama.ai`。

## 2. 下载转写模型（faster-whisper）

faster-whisper 是 Whisper 的 ctranslate2 优化实现，比原版快约 4 倍、省显存。模型从 HuggingFace 下载，放入 `models/` 目录（目录结构须与 `faster-whisper-{档位}/model.bin` 一致）。

```bash
pip install faster-whisper ctranslate2 av

# 方式 A：用 huggingface_hub 下载（推荐）
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-medium', local_dir='models/faster-whisper-medium')"

# 方式 B：国内镜像（hf-mirror）
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-medium', local_dir='models/faster-whisper-medium')"
```

三档可选（`config.json` 里 `whisper_model` 指定路径）：

| 档位 | 大小 | 速度 | 准确率 | 适用 |
|---|---|---|---|---|
| small | ~464MB | 最快 | 中 | 显存/时间紧张 |
| medium | ~1.5GB | 快 | 高 | **推荐默认** |
| large-v3 | ~3GB | 慢 | 最高 | 追求准确率 |

## 3. 安装依赖与配置

```bash
# 创建虚拟环境并安装
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt    # Windows
# .venv/bin/pip install -r requirements.txt      # Linux/macOS

# 生成配置
cp config.example.json config.json
```

`requirements.txt`（核心依赖）：

```
faster-whisper
ctranslate2
PySide6
requests
edge-tts
av
watchdog
huggingface_hub
```

编辑 `config.json`（相对路径基于仓库根目录，也可用绝对路径）：

```jsonc
{
  "inbox_dir": "./inbox",                            // 收件箱（LocalSend 接收目录）
  "vault_dir": "./vault",                            // Obsidian 知识库目录
  "models_dir": "./models",                          // 模型根目录
  "whisper_model": "./models/faster-whisper-medium", // 转写模型路径
  "whisper_compute": "int8",                         // 量化方式
  "ollama_host": "http://127.0.0.1:11434",
  "ollama_model": "qwen2.5:7b",                      // 纪要模型
  "subjects": ["概率论", "离散数学", "电路、信号和系统", "数字逻辑和计算机组成"],  // 固定学科列表
  "supported_ext": [".m4a", ".mp3", ".wav", ".aac", ".flac", ".opus", ".ogg", ".amr", ".wma", ".mp4"]
}
```

## 4. 手机端：LocalSend

- 手机安装 LocalSend（F-Droid / 应用商店）
- 电脑也安装 LocalSend（接收目录设为 `inbox_dir`）
- 录音后用 LocalSend 发送到电脑（局域网直传，不走云端）

## 5. 运行

```bash
python src/app.py
```

- 点击「开始监视」→ 把录音丢进 inbox（或 LocalSend 直传）→ 自动处理
- 处理结果在 Obsidian 中打开 `vault_dir` 浏览
- 应用不常驻、不开机自启；「手动转录」按钮可对任意文件立即转写

## 常见问题

**Q：转写很慢/显存不足？**
改用 small 档（`whisper_model` 指向 small 路径）。

**Q：纪要质量不满意？**
`ollama_model` 换 `qwen2.5:14b`（需 ≥10GB 显存）或 `qwen2.5:3b`（更快）。

**Q：想调整学科列表？**
直接改 `config.json` 的 `subjects` 数组。

**Q：模型下载失败/中断？**
hf-mirror 镜像 + 断点续传（重跑下载命令即可续传）。

**Q：Ollama 弹终端窗口？**
应用已用 `CREATE_NO_WINDOW + SW_HIDE` 拉起 serve；如仍异常，请手动 `ollama serve` 确认可用。
