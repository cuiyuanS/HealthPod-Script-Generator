HealthPod Script Generator (健康播客剧本生成器)

🩺 项目简介

本项目旨在利用大型语言模型（LLM）的能力，将用户上传的体检报告 PDF 转化为结构化、专业且富有幽默感的播客对话脚本。它支持两种不同的运行模式：完全本地化的 VLLM 离线处理模式，以及高效的阿里云 Qwen-Long 云端分析模式。

注意：本项目目前是一个功能演示 (Demo) 脚本集，展示了核心逻辑和 LLM 提示工程，未来可以轻松发展成为一个完整的后端服务（Server），提供体检报告自动播客化的 API 服务。

本项目生成的脚本角色包括：

$$S1$$

 小丽 (主持人, 温暖专业) 和 

$$S2$$

 英俊 (医生朋友, 幽默风趣)，确保输出内容既准确又易于听众理解。

✨ 核心功能

PDF 报告解析： 从复杂的 PDF 体检报告中自动提取关键信息。

结构化分析： 识别患者信息、紧急异常项和重要异常指标。

深度剧本生成： 要求模型对至少 6 项异常指标（如肺部湿罗音、凝血异常等）进行深入的、至少 3 句话的详细解释和生活化比喻。

双模式支持： 提供高性能的本地部署方案（基于 VLLM）和便捷的云端 API 方案（基于阿里云）。

音频合成： 集成专业的 TTS 工具，将剧本转化为可播放的音频。

⚙️ 模式与依赖

本项目提供两套核心脚本，分别针对本地部署和云端服务。

脚本 1: 完全本地化 LLM 管道 (vllm_pipeline.py)

此模式适用于拥有足够计算资源（GPU）的用户，追求数据安全和低延迟的本地处理。

| 组件 | 用途 | 依赖包 |
| PDF 解析 | 从 PDF 中提取原始文本。 | pdfplumber |
| JSON 结构化 | 调用 VLLM 模型，将原始文本总结为结构化 JSON。 | vllm, langchain, pydantic |
| 剧本生成 | 调用 VLLM 模型，根据 JSON 生成对话脚本。 | vllm, langchain |
| 音频合成 (待集成) | 将生成的脚本转换为音频文件。 | SoulX-Podcast (本地 TTS 引擎) |

🛠️ VLLM 设置要求

VLLM API Server： 必须在本地或内网环境中启动 VLLM 的 OpenAI 兼容 API 服务。

# 示例命令 (请替换为您的模型路径和名称)
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct-AWQ --served-model-name qwen2.5 --port 8000



配置文件： 确保脚本中 VLLM_API_BASE 和 VLLM_MODEL 配置正确。

2. 脚本 2: 阿里云 Qwen-Long 分析 (aliyun_qwen_script_generator.py)

此模式利用阿里云的 Qwen-Long 模型强大的文档理解能力，通过文件 ID 直接进行分析和生成，无需本地 GPU 资源。

| 组件 | 用途 | 依赖包 |
| 文件上传 | 将 PDF 报告上传至阿里云存储获取 fileid。 | openai (通过兼容模式) |
| 一站式分析 | 通过 fileid，LLM 直接从 PDF 中提取信息并生成剧本。 | openai (通过兼容模式) |
| 音频合成 (待集成) | 将生成的脚本转换为音频文件。 | CosyVoice (阿里云 TTS 服务) |

🔑 阿里云设置要求

API Key： 在脚本中配置您的 DashScope API Key。

文件上传： 在调用前，您需要通过 DashScope 或 client.files.create 将 PDF 文件上传，获取有效的 fileid。

🚀 快速开始

预置条件

您需要安装 Python 3.8+ 及所需的依赖库：

pip install -r requirements.txt
# 假设 requirements.txt 中包含: openai, langchain, pydantic, pdfplumber, loguru



运行脚本

选择您希望运行的模式，并按照步骤设置环境：

模式 1 (本地 VLLM)

确保您的 VLLM 服务已启动。

将您的 PDF 报告命名为 report.pdf 放在项目根目录。

运行本地脚本：

python vllm_pipeline.py report.pdf



模式 2 (阿里云 Qwen-Long)

在 aliyun_qwen_script_generator.py 中设置您的 API_KEY。

设置 ACTUAL_FILE_ID 为您上传体检报告后获取的阿里云文件 ID。

运行云端脚本：

python aliyun_qwen_script_generator.py



🌟 点击 Start 开始您的体检报告播客化之旅！

🤝 贡献与反馈

欢迎所有形式的贡献！无论是提交 Bug 报告、提出新功能建议，还是直接提交代码，我们都非常期待您的加入。

Fork 本仓库。

创建您的特性分支 (git checkout -b feature/AmazingFeature)。

提交您的修改 (git commit -m 'Add some AmazingFeature')。

推送到分支 (git push origin feature/AmazingFeature)。

打开 Pull Request。

📜 许可证

本项目采用 MIT 许可证 开源。
