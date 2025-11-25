"""
@Project: HealthPod-Script-Generator
@File: healthpod_script_generator.py
@IDE: PyCharm
@Author: CuiYuan
@Date: 25 11月 2025 13:54
@Explain:
一个完全本地化的 LLM 管道，用于自动化生成基于PDF体检报告的播客对话脚本。
工作流程:
1. 通过命令行接收 PDF 报告路径。
2. PDFMedicalReportExtractor 提取核心文本内容。
3. 调用 vLLM (LangChain 结构化输出) 将非结构化文本转换为 JSON 数据。
4. 调用 vLLM (LangChain 文本生成) 根据 JSON 数据生成对话剧本。
5. 打印结果，供后续的 TTS (Text-to-Speech) 服务使用。

【运行环境要求】
1. 确保已安装必要的库: pip install pdfplumber loguru langchain-openai pydantic
2. 确保 vLLM 服务正在本地运行，例如:
   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct-AWQ --served-model-name qwen2.5 --port 8000
   (请根据实际模型和环境调整)

【运行方式】
python healthpod_script_generator.py <your_report.pdf>
"""

import argparse
import asyncio
import json
import re
from typing import Optional, Tuple, List, Dict, Any

import pdfplumber
from loguru import logger
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# --- VLLM/OpenAI API 配置 ---
VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_MODEL = "qwen2.5"
# vLLM/OpenAI 兼容 API 不需要真实的 key
DUMMY_API_KEY = "sk-not-needed"


# --- 1. LLM 客户端初始化 ---
def initialize_llm_client():
    """初始化 LangChain ChatOpenAI 客户端，适配 vLLM API。"""
    logger.info(f"Initializing LLM client with base URL: {VLLM_API_BASE}")
    try:
        llm = ChatOpenAI(
            base_url=VLLM_API_BASE,
            api_key=DUMMY_API_KEY,
            model=VLLM_MODEL,
            temperature=0.7,
            model_kwargs={
                "max_tokens": 4096,  # 增加最大 token 数以支持长剧本
            },
        )
        return llm
    except Exception as e:
        logger.error(f"LLM 客户端初始化失败: {e}")
        raise


# --- 2. PDF 报告提取类 ---
class PDFMedicalReportExtractor:
    """PDF体检报告提取器，专注于提取核心结论和异常项。"""

    def __init__(self):
        """初始化提取器"""
        # 常见目标章节起始关键词 (终检结论, 体检汇总)
        self.start_patterns = [r"终\s*检\s*结\s*论", r"体\s*检\s*汇\s*总"]
        # 常见目标章节结束关键词 (健康建议, 医学解释)
        self.end_patterns = [r"健\s*康\s*建\s*议", r"医\s*学\s*解\s*释"]
        # 需要清除的页眉页脚和干扰信息
        self.patterns_to_remove = [
            r"关注生命、关爱健康",
            r"页码\s*\d+\s*/\s*\d+",
            r"体检号码.*?\n",
            r"本报告仅用于健康检查.*?\n",
        ]

    def normalize_text(self, text: str) -> str:
        """
        深度清洗文本，去除干扰信息和多余空行。
        """
        if not text:
            return ""

        # 去除常见的页眉页脚干扰
        for pattern in self.patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # 修复双重字符 (例如 "终检结结论" -> "终检结论")
        text = re.sub(r"([^\x00-\xff])\1", r"\1", text)

        # 替换多余空格和制表符为单个空格
        text = re.sub(r'\s+', ' ', text)

        # 修正中文句号后的空格
        text = re.sub(r'([。，！？])\s+', r'\1', text)

        # 去除多余空行
        text = re.sub(r'\n{2,}', '\n', text).strip()

        return text

    def _extract_raw_text(self, file_path: str) -> Tuple[str, str]:
        """
        提取PDF中的原始文本，同时区分基础信息和完整文本。
        """
        full_text = ""
        base_info = ""

        try:
            with pdfplumber.open(file_path) as pdf:
                raw_pages = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        if i < 2:
                            # 假设前两页包含基础信息 (姓名、年龄、体检日期等)
                            base_info += text
                        raw_pages.append(text)

                full_text = "\n".join(raw_pages)
        except Exception as e:
            logger.error(f"读取PDF文件失败，请检查文件路径和格式: {e}")
            raise

        return full_text, base_info

    def _find_section(
            self,
            cleaned_text: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        查找目标章节的起始和结束位置。
        """
        start_idx = None
        end_idx = None

        # 寻找开始位置
        for pattern in self.start_patterns:
            match = re.search(pattern, cleaned_text)
            if match:
                start_idx = match.start()
                logger.info(f"找到章节起始点: {match.group()} at {start_idx}")
                break

        # 寻找结束位置，仅在找到起始点后开始搜索
        if start_idx is not None:
            for pattern in self.end_patterns:
                # 在起始点之后的部分搜索
                match = re.search(pattern, cleaned_text[start_idx:])
                if match:
                    # 绝对位置 = 起始搜索点 + 相对匹配位置
                    end_idx = start_idx + match.start()
                    logger.info(f"找到章节结束点: {match.group()} at {end_idx}")
                    break

        return start_idx, end_idx

    def extract(self, file_path: str) -> str:
        """
        核心提取方法：获取 PDF 中的体检报告信息。
        """
        # 提取原始文本
        full_text, base_info = self._extract_raw_text(file_path)

        # 文本清洗
        cleaned_text = self.normalize_text(full_text)
        cleaned_base_info = self.normalize_text(base_info)

        # 查找目标章节位置
        start_idx, end_idx = self._find_section(cleaned_text)

        # 提取总结内容
        if start_idx is not None:
            if end_idx is not None:
                summary_content = cleaned_text[start_idx:end_idx]
            else:
                # 没找到结束词，截取后 2000 个字符作为兜底
                summary_content = cleaned_text[start_idx: start_idx + 2000]
                logger.warning("未找到章节结束点，截取起始点后 2000 个字符作为内容。")
        else:
            logger.warning("未找到章节起始点 ('终检结论')，尝试使用全文前 3000 字作为兜底内容。")
            summary_content = cleaned_text[:3000]

        # 组装最终结果 (用于 LLM 输入)
        final_context = f"""【受检人基础信息】
{cleaned_base_info[:500]}

【核心体检结论与异常项】
{summary_content}"""

        logger.info(f"PDF 文本提取完成，长度: {len(final_context)} 字符。")
        return final_context


# --- 3. Pydantic 结构化输出模型 ---
class AbnormalItem(BaseModel):
    """体检异常指标项"""

    item: str = Field(..., description="异常指标的名称，例如: 收缩压, 胆固醇, 肺部湿罗音。")
    value: str = Field(..., description="异常指标的具体数值或描述，例如: 182, 房颤。")
    flag: str = Field(
        ...,
        description="异常状态的标记，例如: ↑ (高), ↓ (低), 或描述 (例如: 阳性, 异常)。如果无标记，填 '异常' 或 'N/A'。",
    )


class MedicalAnalysis(BaseModel):
    """体检报告结构化分析结果"""

    name: str = Field(..., description="体检者的姓名。")
    gender: str = Field(..., description="体检者的性别，例如: '男' 或 '女'。")
    age: str = Field(..., description="体检者的年龄，包含单位，例如: '45岁'。")
    critical_issues: List[str] = Field(
        ...,
        description="提取出的所有严重或需要立即关注的问题列表，例如: ['房颤', '高血压', '左耳全聋', '乳腺结节']。",
    )
    abnormal_items: List[AbnormalItem] = Field(
        ..., description="所有异常指标项的详细列表 (不包括 critical_issues 中已列出的概括性描述)。"
    )


# --- 4. LLM 交互函数 ---
async def analyze_medical_json(llm_client: ChatOpenAI, text_context: str) -> Dict[str, Any]:
    """步骤1：从文本提取结构化 JSON (使用 LangChain 结构化输出)"""

    logger.info("--- 开始 LLM 步骤 1/2: 结构化数据提取 ---")

    # 1. 定义提示
    system_prompt = f"""
        你是一个专业的医疗数据分析助手。分析提供的体检报告原始文本，并从中提取结构化数据。
        你的任务是准确地提取：姓名、性别、年龄，以及所有主要的严重问题和详细的异常指标项。
        请严格按照给定的 JSON Schema 输出结果，确保所有字段都准确填写。
        """

    try:
        # 使用 with_structured_output 绑定 Pydantic 模型，确保输出格式正确
        structured_llm = llm_client.with_structured_output(MedicalAnalysis)

        # 构造消息
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"请分析这份体检报告文本并提取结构化数据：\n\n{text_context}"
            ),
        ]

        # 调用模型
        analysis_model = await structured_llm.ainvoke(messages)
        logger.success("结构化数据提取成功。")

        # 将 Pydantic Model 转换为 Python Dict
        return analysis_model.model_dump()

    except Exception as e:
        logger.error(f"VLLM 结构化调用失败，请检查 vLLM 服务是否正常运行及 Pydantic 模型是否兼容: {e}")
        raise


async def generate_script(llm_client: ChatOpenAI, analyzed_json: Dict[str, Any]) -> str:
    """步骤2：生成播客剧本 (使用 LangChain Text)"""

    logger.info("--- 开始 LLM 步骤 2/2: 播客剧本生成 ---")

    # 将 JSON 转字符串喂给模型
    json_str = json.dumps(analyzed_json, ensure_ascii=False, indent=2)

    prompt = f"""
    你是由 [S1] 姓名 小丽 (主持人, 温暖专业) 和 [S2] 姓名 英俊 (医生朋友, 幽默, 爱打比方) 组成的播客团队《健康唠嗑》。

    【任务】根据这份体检数据生成对话脚本。请确保剧本内容详实、对话流畅且信息量大，以满足“多说一点”的要求。

    【体检者】{analyzed_json.get('name', 'N/A')} ({analyzed_json.get('gender', 'N/A')}, {analyzed_json.get('age', 'N/A')})

    【体检详情 (JSON)】
    {json_str}

    【要求】
    1. 话题顺序：
       a. 开场：S1 欢迎，S2 打趣。
       b. **核心讨论：** 优先讨论 Critical Issues (严重问题，如房颤、全聋、结节)。
       c. **系统讨论：** 随后讨论 Abnormal Items (其他异常指标，如血压、血脂、肝功能异常等)。
    2. **内容深度 (关键)：**
       a. S2 在解释每个问题时，**必须提到具体指标名称和结果值**，并**进行深入的、至少 3 句话的详细解释**和生活化比喻，确保内容充实，增加剧本长度。
       b. 看到如“乳腺结节”或“肺部结节”等问题时，必须严肃提醒**需要进一步的影像学或病理学排查**。
    3. 结尾：给出 **最少 4 条** 具体、针对性强、可执行的健康建议。
    4. 对话风格：S1 引导话题，S2 负责专业解释和幽默比喻。

    【格式】
    直接输出对话内容，每行以 [S1] 或 [S2] 开头，不要输出Markdown代码块或其他额外说明。
    """

    try:
        # 构造消息
        messages = [
            SystemMessage(
                content="你是专业的播客编剧，只输出对话脚本，不包含任何其他内容。"
            ),
            HumanMessage(content=prompt),
        ]

        # 使用 ainvoke 进行文本生成
        response = await llm_client.ainvoke(messages)
        logger.success("播客剧本生成成功。")
        return response.content.strip()

    except Exception as e:
        logger.error(f"VLLM 剧本生成调用失败: {e}")
        raise


# --- 5. 主程序入口 ---
async def async_main(pdf_path: str):
    """异步主函数，协调 PDF 提取和 LLM 调用。"""

    logger.info(f"--- 启动 HealthPod 剧本生成管道 ---")
    logger.info(f"处理文件: {pdf_path}")

    try:
        # 1. 初始化 LLM 客户端
        llm_client = initialize_llm_client()

        # 2. 提取 PDF 报告文本
        extractor = PDFMedicalReportExtractor()
        text_context = extractor.extract(pdf_path)

        # 3. 结构化分析 (LLM 步骤 1)
        analyzed_data = await analyze_medical_json(llm_client, text_context)

        logger.info("\n--- 结构化数据提取结果 ---")
        print(json.dumps(analyzed_data, indent=2, ensure_ascii=False))

        # 4. 剧本生成 (LLM 步骤 2)
        script = await generate_script(llm_client, analyzed_data)

        logger.info("\n--- 最终生成的播客剧本 ---")
        print(script)
        logger.info("--- 剧本生成完成 ---")

        logger.info("--- SoulX-Podcast 播客音频生成 ---")
        logger.info("--- 参考 https://github.com/Soul-AILab/SoulX-Podcast ---")
        logger.info("--- SoulX-Podcast 播客生成完成 ---")

    except Exception as e:
        logger.error(f"管道执行失败: {e}")
        return


def main():
    """程序命令行入口。"""
    parser = argparse.ArgumentParser(
        description="基于 PDF 体检报告，使用本地 vLLM 服务生成播客对话脚本。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="待处理的体检报告 PDF 文件路径，例如: ./report.pdf"
    )

    args = parser.parse_args()

    # 使用 asyncio.run 运行异步主函数
    asyncio.run(async_main(args.pdf_path))


if __name__ == "__main__":
    main()