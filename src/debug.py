# import pyttsx3

# # 初始化语音引擎
# engine = pyttsx3.init()

# # 获取所有可用的语音
# voices = engine.getProperty('voices')

# # 打印所有可用语音的信息
# for i, voice in enumerate(voices):
#     print(f"语音索引: {i}, 语音 ID: {voice.id}, 语音名称: {voice.name}, 语言: {voice.languages}")

# # 选择一个语音（这里以索引为 1 的语音为例）
# selected_voice = voices[0]

# # 设置语音
# engine.setProperty('voice', selected_voice.id)

# # 要朗读的文本
# text = "这是一个测试语音的示例。"

# # 让引擎朗读文本
# engine.say(text)
# # engine.say("this is a test")
# # 运行引擎并等待朗读完成
# engine.runAndWait()


# import gradio as gr

# def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed):
#     # 这里需要实现具体的音频生成逻辑
#     return "path/to/generated/audio.wav"

# demo = gr.Interface(
#     fn=generate_audio,
#     inputs=[
#         gr.Textbox(label="tts_text"),
#         gr.CheckboxGroup(label="mode_checkbox_group"),
#         gr.Dropdown(label="sft_dropdown"),
#         gr.Textbox(label="prompt_text"),
#         gr.Audio(label="prompt_wav_upload"),
#         gr.Audio(label="prompt_wav_record"),
#         gr.Textbox(label="instruct_text"),
#         gr.Number(label="seed")
#     ],
#     outputs=gr.Audio(label="result_wav_path")
# )

# demo.launch(server_name="127.0.0.1", server_port=50000)
# import sys
# sys.path.append('third_party/Matcha-TTS')
# from CosyVoice.cosyvoice-main.cli.cosyvoice import CosyVoice.cosyvoice as cosyvoice, CosyVoice2
# from cosyvoice.utils.file_utils import load_wav
# import torchaudio
# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# # zero_shot usage
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # bistream usage, you can use generator as input, this is useful when using text llm model as input
# # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# def text_generator():
#     yield '收到好友从远方寄来的生日礼物，'
#     yield '那份意外的惊喜与深深的祝福'
#     yield '让我心中充满了甜蜜的快乐，'
#     yield '笑容如花儿般绽放。'
# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# import gradio as gr

# # 示例 Gradio 应用
# def greet(name):
#     return "Hello " + name + "!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# # 检查是否有类似下面这种自定义路由处理可能影响文件访问
# # 如果有，检查其逻辑是否正确
# # app = demo.launch(server_name="localhost", server_port=8000)
# # @app.get("/gradio_api/file=:file_id/:other_id/:num/playlist.m3u8")
# # def custom_route(file_id, other_id, num):
# #     # 这里可能有错误的权限验证逻辑
# #     return ...

# demo.launch(server_name="localhost", server_port=8000)
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:8000",
#     # 可以根据实际情况添加更多允许的源
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
import subprocess

script1 = 'D:\Aeshtron\CosyVoice\webui.py'
script2 = 'D:\Aeshtron\src\conversation.py'

try:
    process1 = subprocess.Popen(['python', script1])
    process2 = subprocess.Popen(['python', script2])

    # 获取进程返回码
    returncode1 = process1.wait()
    returncode2 = process2.wait()

    if returncode1 == 0 and returncode2 == 0:
        print("两个脚本都成功执行完毕。")
    else:
        print("至少有一个脚本执行过程中出现问题。")

except Exception as e:
    print(f"执行脚本时出错: {e}")