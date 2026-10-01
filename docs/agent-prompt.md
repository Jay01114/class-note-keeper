# 给 AI Agent 的搭建提示词

> 把下面这段提示词整体复制给任意 AI 编程助手 / agent，它就能根据本仓库把整套工作流搭建起来。
> 本仓库只提供工作流与代码，**不捆绑任何商业软件包**；所有软件与模型请按提示词从官方渠道安装。

---

```text
请根据 GitHub 仓库 <https://github.com/Jay01114/class-note-keeper> 中的「课堂笔记管家」项目，在本地把这套"课堂录音自动整理"工作流完整搭建起来并跑通。

## 项目背景
这是一个半自动化的课堂笔记流水线：手机录音 → 电脑自动转写（faster-whisper）→ AI 纪要（Ollama + Qwen2.5）→ 按学科归档进 Obsidian 知识库。全部本地运行、免费。

## 需要你完成的步骤

1. **读代码**：阅读仓库 `src/core.py` 和 `src/app.py`，理解数据流：
   - core.py：配置加载、文件监视、转写、学科识别、纪要生成、归档、去重
   - app.py：PySide6 桌面界面、转写进度条、单实例、Ollama 生命周期管理

2. **安装环境**：
   - 确认 Python 3.11+，创建虚拟环境
   - 安装依赖：faster-whisper、ctranslate2、PySide6、requests、edge-tts、av、watchdog、huggingface_hub

3. **安装 Ollama（从官方渠道）**：
   - 访问 ollama.com 下载对应系统安装包，或按其官方脚本安装
   - 拉取纪要模型：`ollama pull qwen2.5:7b`（显存小可换 3b；追求质量可 14b）

4. **下载转写模型（faster-whisper，HuggingFace 官方）**：
   - 仓库根目录建 `models/`，用 huggingface_hub 下载 `Systran/faster-whisper-medium` 到 `models/faster-whisper-medium`
   - 国内网络用镜像：设置环境变量 HF_ENDPOINT=https://hf-mirror.com 后重试
   - 可选 small（快）/ large-v3（准）

5. **配置**：复制 `config.example.json` 为 `config.json`，按本机实际情况修改：
   - inbox_dir / vault_dir / models_dir 换成你自己的路径
   - subjects 改成你的学科列表（或保留默认）

6. **运行验证**：
   - `python src/app.py` 启动界面
   - 点「开始监视」，向 inbox 目录放一段测试音频（可用 edge-tts 生成：`edge-tts --voice zh-CN-XiaoxiaoNeural --text "测试" --write-media test.mp3`）
   - 确认：自动转写 → 学科识别正确 → 归档到 vault/学科/纪要/ 目录

7. **收尾**：
   - 把完成的搭建步骤、使用的路径、验证结果汇总给我
   - 如果某一步失败，把完整报错贴出来，分析根因后重试

## 注意事项
- 所有软件与模型从官方渠道安装，不要安装第三方打包的"整合包"
- 本工作流定位"晚间批量整理"：不常驻后台、不开机自启
- 遇到 Ollama 相关报错，先确认 `curl http://127.0.0.1:11434/api/version` 能返回版本号
```

---

## 为什么这样写

- **明确"从官方渠道安装"**：仓库不捆绑软件包，避免商用/侵权问题，只提供编排思路；
- **给 agent 一条可验证的闭环**：装依赖 → 装 Ollama → 下模型 → 配置 → 跑测试音频 → 确认归档，每一步都有预期结果；
- **容错指引**：国内镜像、Ollama 健康检查，都是真实踩过的坑。
