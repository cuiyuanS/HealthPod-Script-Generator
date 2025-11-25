# 🩺 HealthPod Script Generator (健康播客剧本生成器)

> 让冰冷的体检报告，变成温暖易懂的播客对话 ✨

## 项目简介

本项目旨在利用大型语言模型（LLM）的强大能力，将用户上传的体检报告 PDF 转化为 **结构化、专业且富有幽默感** 的播客对话脚本。支持两种灵活运行模式，兼顾数据安全与便捷性，轻松实现体检报告的 “音频化解读”。

⚠️ 项目定位：当前为功能演示（Demo）脚本集，展示核心逻辑与 LLM 提示工程，未来可无缝扩展为完整后端服务，提供 API 接口支持。

### 固定角色设定

- **小丽（主持人）**：温暖亲和，专业严谨，擅长引导话题、衔接内容

- **英俊（医生朋友）**：幽默风趣，深入浅出，用生活化语言解读专业医学指标

## ✨ 核心功能

| 功能模块       | 详细说明                                                     |
| -------------- | ------------------------------------------------------------ |
| 📄 PDF 报告解析 | 智能提取复杂 PDF 中的关键医疗信息，剔除冗余内容              |
| 📊 结构化分析   | 精准识别患者基础信息、**紧急异常项**、重要异常指标           |
| 🎭 深度剧本生成 | 对 ≥6 项异常指标（如肺部湿罗音、凝血异常等）提供：・至少 3 句话的专业解读・生活化比喻辅助理解・双角色自然对话互动 |
| ⚡ 双模式支持   | 本地部署（VLLM）：数据安全、低延迟云端服务（阿里云 Qwen-Long）：无需 GPU、高效便捷 |
| 🔊 音频合成     | 集成专业 TTS 工具，一键将剧本转化为可播放音频  |

## 🎧 示例音频预览

<table>
<tr>
<td align="center">

**audio1**

</td>
<td align="center">

**audio2**

</td>
</tr>
<tr>
<td align="center">

[audio1.webm](https://github.com/cuiyuanS/HealthPod-Script-Generator/releases/download/untagged-11f3d45d97c49f4ae7fd/audio1.webm)

</td>
<td align="center">

[audio2.webm](https://github.com/cuiyuanS/HealthPod-Script-Generator/releases/download/untagged-11f3d45d97c49f4ae7fd/audio2.webm)
</td>
</tr>
</table>
</details>



## ⚙️ 运行模式与依赖

### 模式 1：完全本地化 LLM 管道（[vllm_pipeline.py](vllm_pipeline.py)）

适用于拥有充足 GPU 资源、注重数据隐私的用户，实现全流程离线处理。

| 组件             | 用途                              | 核心依赖包                     |
| ---------------- | --------------------------------- | ------------------------------ |
| PDF 解析         | 提取 PDF 原始文本内容             | pdfplumber                     |
| JSON 结构化      | 调用 VLLM 生成标准化医疗信息 JSON | vllm, langchain, pydantic      |
| 剧本生成         | 基于 JSON 生成双角色对话脚本      | vllm, langchain                |
| 音频合成 | 脚本转音频文件                    | SoulX-Podcast（本地 TTS 引擎） |

#### 📌 VLLM 环境设置

1. **启动 VLLM API Server**（本地 / 内网环境）：

```
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct-AWQ \
  --served-model-name qwen2.5 \
  --port 8000
```

> 请替换为实际的模型路径、名称和端口

1. **配置验证**：确保脚本中 VLLM_API_BASE 和 VLLM_MODEL 参数与服务一致

#### ⚠️ 注意

如需使用本地音频合成功能，请先安装：[SoulX-Podcast GitHub 仓库](https://github.com/Soul-AILab/SoulX-Podcast)

### 模式 2：阿里云 Qwen-Long 分析（[aliyun_qwen_script_generator.py](aliyun_qwen_script_generator.py)）

利用阿里云 Qwen-Long 模型的强大文档理解能力，通过文件 ID 直接分析，无需本地 GPU 资源，快速生成剧本。

| 组件               | 用途                                   | 核心依赖包                   |
| ------------------ | -------------------------------------- | ---------------------------- |
| 文件上传           | 上传 PDF 至阿里云存储，获取唯一 fileid | openai（兼容模式）           |
| 一站式分析         | 基于 fileid 提取信息 + 生成剧本        | openai（兼容模式）           |
| 音频合成（待集成） | 脚本转音频文件                         | CosyVoice（阿里云 TTS 服务） |

#### 📌 阿里云环境设置

1. **API Key 配置**：在脚本中填入您的 DashScope API Key

1. **文件上传**：先通过 DashScope 或 client.files.create 上传 PDF，获取有效 fileid

## 🚀 快速开始

### 前置条件

- Python 3.10+ 环境

- 安装依赖库：

```
pip install -r requirements.txt
```

> requirements.txt 包含：openai, langchain, pydantic, pdfplumber, loguru

### 模式 1：本地 VLLM 运行步骤

1. 启动本地 VLLM API Server（参考前文命令）

1. 将体检报告 PDF 命名为 report.pdf，放入项目根目录

1. 运行脚本：

```
python vllm_pipeline.py report.pdf
```

### 模式 2：阿里云 Qwen-Long 运行步骤

1. 打开 [aliyun_qwen_script_generator.py](http://aliyun_qwen_script_generator.py)，配置：

- - API_KEY：您的 DashScope API Key

- - ACTUAL_FILE_ID：上传 PDF 后获取的阿里云文件 ID

1. 运行脚本：

```
python aliyun_qwen_script_generator.py
```

## 🤝 贡献与反馈

欢迎任何形式的贡献！无论是 Bug 报告、功能建议还是代码提交，我们都非常期待您的参与～

1. Fork 本仓库

1. 创建特性分支：git checkout -b feature/AmazingFeature

1. 提交修改：git commit -m 'Add some AmazingFeature'

1. 推送到分支：git push origin feature/AmazingFeature

1. 打开 Pull Request

## 📜 许可证

本项目采用 **MIT 许可证** 开源，详见 [LICENSE](LICENSE) 文件。

> 💡 提示：如果需要扩展功能（如多语言支持、自定义角色人设、异常指标预警等级配置），可以提交 Issue 或联系维护者～
