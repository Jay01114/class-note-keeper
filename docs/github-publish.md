# 发布到 GitHub：建仓资料与填表规范

## 一、创建仓库（github.com/new）

GitHub 新建仓库页需要填的内容，直接照抄：

| 字段 | 建议填写 |
|---|---|
| **Repository name** | `class-note-keeper`（或 `lecture-note-automator`） |
| **Description** | `Local-first lecture note pipeline: phone recording → faster-whisper transcription → Ollama/Qwen summary → Obsidian vault. 本地优先的课堂录音自动整理流水线。` |
| **Public** | 选 **Public**（公开） |
| **Add a README** | 勾选（仓库会自动生成，但我们会用仓库里的 README.md 覆盖） |
| **Add .gitignore** | 选 **Python** |
| **Choose a license** | 选 **MIT License** |

创建后，把 `opensource/` 目录下的所有文件（README.md、LICENSE、.gitignore、config.example.json、src/、docs/）上传覆盖即可。

## 二、关于"语音识别没成功"

如果你在 GitHub 建仓时遇到填写"语音识别"相关内容不成功，最常见原因是：

1. **仓库名不能用中文** —— GitHub 的 Repository name 只支持英文小写、数字、连字符（`-`）、下划线。`语音识别` 这样的中文名会直接报错。→ 用英文名（如上面的 `class-note-keeper`）。
2. **Description / README 内容里可以写中文** —— 描述、文档用中文完全没问题。
3. **Topics（主题标签）建议用英文** —— GitHub 的 Topics 虽然支持部分 Unicode，但中文标签很难被检索到，且不被官方收录。→ 用英文标签（见下表）。

## 三、Topics 标签（英文，公开后可自行编辑）

```
whisper
faster-whisper
ollama
qwen
speech-to-text
transcription
obsidian
note-taking
local-ai
offline
pyside6
desktop-app
```

> 添加位置：仓库首页 → 右侧 "About" 区域 → ⚙ 设置 → Topics。

## 四、发布前检查清单

- [ ] `README.md` 已更新（README 里的"仓库地址"占位符换成真实地址）
- [ ] `docs/agent-prompt.md` 里的 `<此处填仓库地址>` 已替换
- [ ] `config.json`（含本机路径的）**没有**被提交（.gitignore 已排除；只提交 `config.example.json`）
- [ ] `models/`、`inbox/`、`vault/` 等本地数据目录未被提交
- [ ] LICENSE 已选 MIT（README 底部有对应徽章可加）
- [ ] 代码中无个人敏感信息（用户名、机器路径等——本项目已清理）

## 五、可选：README 徽章

发布后可在 README 顶部加徽章（用 shields.io）：

```markdown
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/github/license/<你的用户名>/class-note-keeper)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
```

## 六、补充说明

- **为什么不捆绑软件包**：Ollama、LocalSend 等是第三方软件/服务，各自有许可证；本项目只做"编排"（用它们的公开接口），不打包、不重分发，避免许可证与商用纠纷。搭建方式全部写在 `docs/setup-guide.md`。
- **模型文件不入库**：模型权重（数百 MB ~ 数 GB）应通过 HuggingFace 官方下载（见 setup-guide.md），不要提交到仓库。
