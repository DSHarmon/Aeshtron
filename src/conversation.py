import datetime
import ollama
from playsound import playsound

# from ASR.asr_interface import ASRInterface
from ASR.ASR import VoiceRecognition
from TTS.pyttxs3_TTS import TTSEngine
from loguru import logger

# 从 CosyVoice 中导入 CosyVoice 类
import sys
import os
import torchaudio
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录
target_dir = os.path.abspath(os.path.join(current_dir, '..', 'CosyVoice'))
# 将上级目录添加到 sys.path 中
sys.path.append(target_dir)

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav


prompttext = "你好，我是Aeshtron。"


if __name__ == "__main__":
    # 创建 VoiceRecognition 类的实例
    vr = VoiceRecognition(
        model_type="sense_voice",
        sense_voice="./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx",
        tokens="./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
        num_threads=4,
        use_itn=True,
        provider="cpu"
    )

# 打开文件以追加模式写入聊天记录
with open('./logs/chat_history.txt', 'a', encoding="utf-8") as file:
    while True:
        # 用户选择输入方式
        host="127.0.0.1"
        port="11434"
        client= ollama.Client(host=f"http://{host}:{port}")

        user_input = input(">> ")
        if user_input.lower() == '':
            transcription = vr.transcribe_from_mic_with_vad()
            print("识别结果:", transcription)
            user_input = transcription
    
        # 记录当前时间
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 将用户输入写入文件
        file.write(f"{current_time} - 用户: {user_input}\n")

        # 调用 Ollama 获取流式回复
        full_response = ""
        print("Aeshtron: ", end="", flush=True)
        for part in client.generate(model="qwen2.5:latest", prompt=user_input, stream=True):
            response_part = part["response"]
            full_response += response_part
            print(response_part, end="", flush=True)
        print()  # 换行
        file.write(f"{current_time} - Aeshtron: {full_response}\n")
        file.write("-" * 50 + "\n")

        # 使用 CosyVoice 生成音频
        from gradio_client import Client, handle_file
        prompt_wavdir = 'D:/Aeshtron/src/TTS/prompt_wav/test.wav'    
        client = Client("http://localhost:8000/")
        result = client.predict(
                tts_text=full_response,
                mode_checkbox_group="3s极速复刻",
                prompt_text=prompttext,
                prompt_wav_upload=handle_file('D:/Aeshtron/src/TTS/prompt_wav/test.wav'),
                prompt_wav_record=None,#handle_file('D:/Aeshtron/src/TTS/prompt_wav/test.wav'),
                instruct_text="",
                seed=0,
                stream=False,
                speed=1,
                api_name="/generate_audio"
        )
        print(result)

    
        # 指定临时目录
        wav_files = []
        seekfiledir = 'C:/Users/DSHarmon/AppData/Local/Temp/gradio'
        # 遍历指定目录及其子目录
        for root, _, files in os.walk(seekfiledir):
            for seekfile in files:
                if seekfile.endswith('.wav'):
                    file_path = os.path.join(root, seekfile)
                    wav_files.append((file_path, os.path.getmtime(file_path)))

        # 如果没有找到 .wav 文件
        if not wav_files:
            print("未找到 .wav 文件。")
            

        # 按修改时间排序，获取最新的文件
        wav_files.sort(key=lambda x: x[1], reverse=True)
        latest_file = wav_files[0][0]

        try:
            print(f"找到最新的音频文件: {latest_file}")
            # 播放音频文件
            playsound(latest_file)
        except Exception as e:
            print(f"播放音频文件时出错: {e}")

            # 使用 TTS 引擎生成音频并播放    
            # tts_engine = TTSEngine()
            # audio_file = tts_engine.generate_audio(full_response)
            # print(f"生成的音频文件路径: {audio_file}")
            # try:playsound(audio_file)
            # except Exception as e:
            #     logger.error(f"播放音频时出错: {e}")

        # 将完整回复写入文件

        
