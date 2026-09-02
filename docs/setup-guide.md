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
json5
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
  "supported_ext": [".m4a", ".mp3", ".wav", ".aac", ".flac", ".opus", ".ogg", ".amr", ".wma", ".mp4"],

  // 知识补全（[补] 标记）：纪要归档后，7B 检测转写里讲得含糊/缺失的知识点，
  // 高置信候选经维基百科查证后内嵌补回纪要。全免费无 key。
  "knowledge": {
    "enabled": true,
    "max_gap_items": 5,
    "max_sentences": 4,
    "search_backend": "wikipedia",
    "wiki_lang": "zh",
    "wiki_endpoint": "https://proxy.littlejoy.live/api/wikipedia/zh", // 国内可达的维基 API 代理；海外可直接用 https://zh.wikipedia.org/w/api.php
    "wiki_proxy": "",                                 // 如需走代理查维基可在此填 http://...（留空直连）
    "wiki_unreachable_policy": "high_only",           // 维基不可达时：仅高置信 7B 候选落盘
    "mark_patched": true
  },

  // 术语纠错表（语音识别谐音误写 → 标准术语，按学科分组）。
  // 只放"标准语境下几乎必为误写"的高置信词；生成纪要前后各自动替换一次。
  "term_fixes": {
    "离散数学": [
      ["柔耻原理", "容斥原理"], ["柔齿原理", "容斥原理"], ["满色", "满射"], ["单色", "单射"], ["双色", "双射"]
    ],
    "概率论": [
      ["克鲁姆克洛夫", "柯尔莫哥洛夫"], ["刺客家境", "数学归纳法"]
    ]
  }
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

**Q：2 小时以上的长录音会丢内容吗？**
不会。纪要采用「分段整理 + 程序拼接」：转写按 12000 字符切成多段，每段独立交给 7B 整理成草稿，再由程序按小节归并、条目去重后拼成整门课纪要。不依赖模型一次性读完全文，已有内容零丢失（旧版只喂前 2 万字符导致后半截被丢弃的问题已修复）。

**Q：纪要里 `[补]` 标记是什么？**
知识补全功能（`knowledge` 配置）。AI 纪要里原文讲不全或录音听不清的知识点，会以 `[补]` 前缀内嵌补回正确内容；补全候选需经维基百科查证，查证不可达时宁缺毋滥。不想要可设 `knowledge.enabled: false`。

**Q：识别出来的术语是谐音错字（如"容赤原理"）？**
语音识别对专业术语常产生谐音误写。可在 `term_fixes` 对应学科下加 `["错误写法", "正确写法"]`，生成纪要前后会自动整词替换（只替换表内高置信词，对正确内容无副作用）。
