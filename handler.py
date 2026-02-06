import base64
import os
import runpod

from tts_generator import get_tts_generator


def handler(event):
    """
    RunPod Serverless handler

    请求示例:
    {
        "input": {
            "text": "要转换的一句话"
        }
    }

    返回:
    {
        "audio_format": "wav",
        "audio_base64": "<base64 编码音频>"
    }
    """
    print("Worker Start")
    data = event.get("input") or {}

    # 支持 text 和 prompt 两种字段名
    text = data.get("text") or data.get("prompt")
    if not text or not str(text).strip():
        return {"error": "text 不能为空，请在 input.text 或 input.prompt 中传入一句文字"}

    # 语音参考音色文件路径，通过环境变量配置，默认使用项目根目录下的 voice_reference.wav
    voice_reference = os.environ.get("VOICE_REFERENCE_PATH", "voice_reference.wav")

    # 输出临时文件
    output_path = "/tmp/output.wav"

    # 获取单例 TTS 生成器（内部会加载 IndexTTS2 模型）
    tts = get_tts_generator()

    # 生成语音（这里使用 wav，避免对 ffmpeg 的依赖）
    tts.generate(
        text=text,
        voice_reference=voice_reference,
        output_path=output_path,
        output_format="wav",
    )

    # 读取生成的音频并做 base64 编码返回
    with open(output_path, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "audio_format": "wav",
        "audio_base64": audio_b64,
    }


# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
