# 安装与配置

3.1.0 是供使用者自行配置环境的源码版本，不包含 Python、Ollama 或模型权重。需要 Windows、Python 3.11+、Ollama、Qwen 模型和 faster-whisper 模型文件。

1. 安装 Python 3.11 或更新的兼容版本，并安装 Ollama。
2. 安装项目依赖：

   ```bash
   python -m pip install -r requirements.txt
   ```

3. 下载纪要模型：

   ```bash
   ollama pull qwen2.5:7b
   ```

4. 准备 faster-whisper 模型文件，并按需更新 `config.example.json` 中的模型、收件箱、Obsidian vault 和 Ollama 地址。复制配置为 `config.json`，不要把个人目录或录音提交到仓库。
5. 启动：

   ```bash
   python src/app.py
   ```

应用监视配置的 inbox，等音频文件大小稳定后开始处理。转写原文、会议纪要和录音归档到 `vault/会议/{转写,纪要,原材料}`。

## 文件时间提示

归档文件名采用 `YYYY-MM-DD_HH-mm_会议主题`。程序先尝试解析源文件名中的日期时间，解析不到时读取文件修改时间。时间来源写入 `.meta.json`。录音文件名时间或文件修改时间不一定等于会议开始时间，请以会议实际信息为准。

## 运行环境

Windows 上转写默认使用 CUDA。若显卡或 CUDA 不可用，应按 faster-whisper 官方说明调整设备配置。Ollama 服务须由使用者安装并运行；模型缺失、配置路径无效或服务不可用时，处理任务会失败并记录日志。
