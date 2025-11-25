"""
@Project: HealthPod-Script-Generator
@File: aliyun_qwen_script_generator.py
@IDE: PyCharm
@Author: CuiYuan
@Date: 25 11月 2025 13:55
@Explain:
基于阿里云 Qwen-Long 模型的文件分析能力，生成体检报告播客对话剧本，并使用 CosyVoice 语音合成服务生成音频文件。

【运行环境要求】
1. 确保已安装必要的库: pip install openai dashscope loguru pydantic argparse
   - 注意：`dashscope` 用于 TTS，`openai` 用于 Qwen-Long (兼容模式)。
2. 确保系统中已安装 `ffmpeg` (用于合并生成的音频片段)。

【配置说明】
请将 API_KEY 替换为您在阿里云获取的真实 API Key。

【运行方式】
python aliyun_qwen_script_generator.py <your_report.pdf>
"""
import argparse
import asyncio
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, List, Optional

# 阿里云 SDK
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer
from openai import OpenAI, BadRequestError
from loguru import logger

# --- Qwen-Long API 配置 ---
# !!! 请替换为您在阿里云获取的真实 API Key !!!
API_KEY = ""
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-long"

dashscope.api_key = API_KEY


# --- 1. Aliyun 客户端初始化 ---
def initialize_aliyun_client() -> OpenAI:
    """初始化 OpenAI 客户端，以兼容模式适配阿里云 DashScope API。"""
    if not API_KEY:
        logger.error("API_KEY 未设置。请在脚本中填入您的真实 API Key。")
        raise ValueError("API_KEY is not configured.")

    logger.info(f"Initializing Aliyun client for model: {MODEL_NAME}")
    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        return client
    except Exception as e:
        logger.error(f"Aliyun 客户端初始化失败: {e}")
        raise


# --- 2. 文件上传与 LLM 分析 ---

def upload_file_to_aliyun(aliyun_client: OpenAI, file_path: str) -> str:
    """
    上传文件到阿里云，并返回文件 ID。
    :param aliyun_client: 阿里云客户端
    :param file_path: 本地 PDF 文件路径
    :return: 上传后的文件 ID
    """
    logger.info(f"Uploading file for analysis: {file_path}")
    try:
        file_object = aliyun_client.files.create(
            file=Path(file_path),
            purpose="file-extract"
        )
        logger.success(f"File uploaded successfully. File ID: {file_object.id}")
        return file_object.id
    except Exception as e:
        logger.error(f"上传文件到阿里云失败: {e}")
        raise


def _generate_qwen_script_prompt() -> str:
    """
    生成用于 Qwen-Long 模型的超级提示词，指导模型完成分析和剧本生成。
    """
    data_section = "体检报告已通过文件 ID 导入，请基于此报告内容进行分析。"

    prompt = f"""
你是由 [S1] 姓名 小丽 (主持人, 温暖专业) 和 [S2] 姓名 英俊 (医生朋友, 幽默, 爱打比方) 组成的播客团队《健康唠嗑》。

【任务】根据{data_section}，生成一期详尽且生动的对话脚本。请确保剧本内容详实、对话流畅，信息量大，S2的解释必须深入到位，满足以下所有要求。

【指令】
1. **信息提取**：首先从体检报告中识别出体检者的姓名、性别和年龄，并在开场白中提及。
2. **开场**：S1 欢迎，S2 打趣。

【讨论核心要点及要求】
1. **紧急问题（开场即讨论）：** 心脏律不齐（房颤倾向），右耳全聋。请模型从报告中准确提取相关异常项。
2. **至少 6 项重要异常项（需深入讨论）：** - 肺部：右肺可闻及散在湿罗音
   - 肝脏：轻度肿大
   - 甲状腺：甲状腺Ⅰ度肿大
   - 乳腺：乳腺结节
   - 前列腺：前列腺Ⅰ度增生
   - 血液指标异常：血浆凝血酶原时间测定(偏高) 和 血小板分布宽度PDW(偏低)。

3. **S2 解释深度要求：**
   - 针对每一项异常，S2 必须明确说出指标名称和具体的异常描述。
   - S2 必须进行深入的、**至少 3 句话**的详细解释，并使用生动的生活化比喻。
4. **特殊提醒**：必须严肃提醒：乳腺结节（即使是男性报告中出现也需重视）需要立即进行进一步影像学检查排查。

【结尾建议】
必须给出 **最少 4 条** 具体、可执行、针对性的健康建议。

【格式要求】
直接输出对话内容，每行以 [S1] 或 [S2] 开头，不要输出Markdown代码块、标题或其他额外说明。
"""
    return prompt


def get_ai_result_by_fileid(aliyun_client: OpenAI, fileid: str) -> str:
    """
    通过 fileid 调用 Qwen-Long 模型进行分析和剧本生成。
    :param aliyun_client: 阿里云客户端
    :param fileid: 上传到阿里云后的文件 ID。
    :return: 生成的剧本内容。
    """
    logger.info(f"--- 方案一: 基于文件 ID ({fileid}) 调用 Qwen-Long 模型 ---")

    try:
        user_prompt = _generate_qwen_script_prompt()

        # 构造 Messages，将 fileid:// 放在 content 中
        messages = [
            {'role': 'system',
             'content': '你是一位专业的播客编剧和医疗数据分析专家，只输出对话脚本，不包含任何其他内容。'},
            {'role': 'user', 'content': f'fileid://{fileid}\n\n{user_prompt}'}
        ]

        completion = aliyun_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True}
        )

        full_content = ""
        logger.info("--- 剧本生成中 (实时输出) ---")
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                # 实时打印输出
                print(content, end="", flush=True)

            if chunk.usage:
                # 打印总计 token 使用量
                logger.info(f"\n[Token 使用情况]: 总计 tokens: {chunk.usage.total_tokens}")

        logger.success("Qwen-Long 剧本生成成功。")
        return full_content.strip()

    except BadRequestError as e:
        logger.error(
            f"\nQwen-Long API 调用失败。请检查 API Key、文件 ID 和模型权限。"
        )
        logger.error(f"原始错误信息：{e}")
        return ""
    except Exception as e:
        logger.error(f"\n发生未知错误：{e}")
        return ""


# --- 3. CosyVoice 语音合成 ---

def split_text_by_roles(text: str) -> List[Tuple[str, str]]:
    """分割文本为角色和内容的列表 (S1/S2)。"""
    lines = text.strip().split('\n')
    segments = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('[S1]'):
            content = line.replace('[S1]', '').strip()
            segments.append(('S1', content))
        elif line.startswith('[S2]'):
            content = line.replace('[S2]', '').strip()
            segments.append(('S2', content))

    return segments


def synthesize_speech(role: str, text: str, index: int) -> Tuple[int, Optional[str]]:
    """
    使用 CosyVoice 合成语音。
    :return: (片段索引, 临时文件路径)
    """
    try:
        # 配置不同的音色
        if role == 'S1':
            voice = "longanling_v2"  # 女声, 温暖专业
        else:  # S2
            voice = "longanzhi_v3"  # 男声, 幽默医生

        synthesizer = SpeechSynthesizer(
            model="cosyvoice-v3-flash",
            voice=voice,
            instruction="播客风格，口语化表达，自然停顿"
        )

        # 合成语音
        audio_data = synthesizer.call(text)

        # 保存临时文件
        temp_filename = f'temp_audio_{index}_{role}.mp3'
        with open(temp_filename, 'wb') as f:
            f.write(audio_data)

        # 打印指标信息
        # print(f'[Metric] {role} 片段 {index} requestId为：{synthesizer.get_last_request_id()}，首包延迟为：{synthesizer.get_first_package_delay()}毫秒')
        logger.debug(f"成功合成 {role} 片段 {index} 到 {temp_filename}")

        return (index, temp_filename)

    except Exception as e:
        logger.error(f"合成 {role} 片段 {index} 时出错：{e}")
        return (index, None)


def merge_audio_files(segments_info: List[Tuple[int, str]], output_path: str):
    """
    使用 ffmpeg 合并音频文件。
    """
    try:
        import os

        # 排序确保片段顺序正确
        sorted_segments = sorted(segments_info)

        # 创建文件列表
        file_list = []
        for idx, filename in sorted_segments:
            if filename and os.path.exists(filename):
                file_list.append(f"file '{os.path.abspath(filename)}'") # 使用绝对路径更安全

        # 创建临时文件列表
        with open('file_list.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(file_list))

        # 使用ffmpeg合并
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', 'file_list.txt', '-c', 'copy', output_path
        ]

        logger.info(f"Merging {len(file_list)} audio files to {output_path} using ffmpeg...")
        subprocess.run(cmd, check=True, capture_output=True)

        logger.success(f"音频已合并并保存到: {output_path}")

        # 清理临时文件
        os.remove('file_list.txt')
        for _, filename in segments_info:
            if filename and os.path.exists(filename):
                os.remove(filename)
        logger.info("临时文件清理完成。")

    except FileNotFoundError:
        logger.error("ffmpeg 未安装或不在系统路径中。请确保已安装 ffmpeg！")
        raise
    except Exception as e:
        logger.error(f"合并音频时出错: {e}")
        raise


def aliyun_cosyvoice_tts(script: str, output_path: str):
    """协调 CosyVoice 语音合成和音频合并流程。"""

    # 分割文本
    segments = split_text_by_roles(script)
    if not segments:
        logger.warning("剧本内容为空或无法识别角色，跳过语音合成。")
        return

    logger.info(f"共分割出 {len(segments)} 个语音片段，开始并发合成...")

    # 使用线程池并发合成
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for i, (role, text) in enumerate(segments):
            if text:
                future = executor.submit(synthesize_speech, role, text, i)
                futures.append(future)

        # 收集结果
        results = []
        for future in futures:
            result = future.result()
            if result and result[1]:  # 如果成功生成了音频
                results.append(result)

    # 合并音频
    if results:
        merge_audio_files(results, output_path)
    else:
        logger.error("没有成功生成任何音频文件。")


# --- 4. 主程序入口 ---
async def async_main(pdf_path: str):
    """异步主函数，协调文件上传、LLM 调用和 TTS 合成。"""

    logger.info(f"--- 启动 HealthPod Aliyun 剧本生成管道 ---")
    logger.info(f"处理文件: {pdf_path}")

    aliyun_client = None
    file_id = None
    script = ""

    try:
        # 1. 初始化 aliyun 客户端
        aliyun_client = initialize_aliyun_client()

        # 2. 文件上传
        file_id = upload_file_to_aliyun(aliyun_client, pdf_path)

        # 3. 剧本生成 (LLM)
        script = get_ai_result_by_fileid(aliyun_client, file_id)

        # 4. 语音合成 (TTS)
        if script:
            output_path = f"{Path(pdf_path).stem}_podcast.mp3"
            aliyun_cosyvoice_tts(script, output_path)

    except Exception as e:
        logger.error(f"管道执行失败: {e}")
        # 在 finally 中处理文件清理

    finally:
        # 5. 清理已上传的文件
        if aliyun_client and file_id:
            try:
                # aliyun_client.files.delete(file_id) # 兼容模式下的文件删除接口可能有所不同，这里注释掉，避免报错
                logger.warning(f"请手动清理上传的文件ID: {file_id} 以节省资源。")
            except Exception as e:
                logger.warning(f"文件清理失败: {e}")

        logger.info("--- 管道执行结束 ---")


def main():
    """程序命令行入口。"""
    # 配置 loguru
    logger.add("file_{time}.log", rotation="10 MB", level="INFO")

    parser = argparse.ArgumentParser(
        description="基于 PDF 体检报告，阿里云 Qwen-Long 分析服务生成播客对话脚本。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="待处理的体检报告 PDF 文件路径，例如: ./report.pdf"
    )

    args = parser.parse_args()

    # 检查 PDF 文件是否存在
    if not os.path.exists(args.pdf_path):
        logger.error(f"文件未找到: {args.pdf_path}")
        return

    # 使用 asyncio.run 运行异步主函数
    asyncio.run(async_main(args.pdf_path))


if __name__ == "__main__":
    main()