import os
import numpy as np
import sherpa_onnx
from loguru import logger
from .asr_interface import ASRInterface
from .utils import download_and_extract
import onnxruntime
import pyaudio
import webrtcvad

class VoiceRecognition(ASRInterface):
    def __init__(
        self,
        model_type: str = "paraformer",  # or "transducer", "nemo_ctc", "wenet_ctc", "whisper", "tdnn_ctc", "sense_voice"
        encoder: str = None,  # Path to the encoder model, used with transducer
        decoder: str = None,  # Path to the decoder model, used with transducer
        joiner: str = None,  # Path to the joiner model, used with transducer
        paraformer: str = None,  # Path to the model.onnx from Paraformer
        nemo_ctc: str = None,  # Path to the model.onnx from NeMo CTC
        wenet_ctc: str = None,  # Path to the model.onnx from WeNet CTC
        tdnn_model: str = None,  # Path to the model.onnx for the tdnn model of the yesno recipe
        whisper_encoder: str = None,  # Path to whisper encoder model
        whisper_decoder: str = None,  # Path to whisper decoder model
        sense_voice: str = None,  # Path to the model.onnx from SenseVoice
        tokens: str = None,  # Path to tokens.txt
        hotwords_file: str = "",  # Path to hotwords file
        hotwords_score: float = 1.5,  # Hotwords score
        modeling_unit: str = "",  # Modeling unit for hotwords
        bpe_vocab: str = "",  # Path to bpe vocabulary, used with hotwords
        num_threads: int = 1,  # Number of threads for neural network computation
        whisper_language: str = "",  # Language for whisper model
        whisper_task: str = "transcribe",  # Task for whisper model (transcribe or translate)
        whisper_tail_paddings: int = -1,  # Tail padding frames for whisper model
        blank_penalty: float = 0.0,  # Penalty for blank symbol
        decoding_method: str = "greedy_search",  # Decoding method (greedy_search or modified_beam_search)
        debug: bool = False,  # Show debug messages
        sample_rate: int = 16000,  # Sample rate
        feature_dim: int = 80,  # Feature dimension
        use_itn: bool = True,  # Use ITN for SenseVoice models
        provider: str = "cpu",  # Provider for inference (cpu or cuda)
    ) -> None:
        self.model_type = model_type
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.paraformer = paraformer
        self.nemo_ctc = nemo_ctc
        self.wenet_ctc = wenet_ctc
        self.tdnn_model = tdnn_model
        self.whisper_encoder = whisper_encoder
        self.whisper_decoder = whisper_decoder
        self.sense_voice: str = sense_voice
        self.tokens = tokens
        self.hotwords_file = hotwords_file
        self.hotwords_score = hotwords_score
        self.modeling_unit = modeling_unit
        self.bpe_vocab = bpe_vocab
        self.num_threads = num_threads
        self.whisper_language = whisper_language
        self.whisper_task = whisper_task
        self.whisper_tail_paddings = whisper_tail_paddings
        self.blank_penalty = blank_penalty
        self.decoding_method = decoding_method
        self.debug = debug
        self.SAMPLE_RATE = sample_rate
        self.feature_dim = feature_dim
        self.use_itn = use_itn

        # we need to find a way to get cuda version of sherpa-onnx before we can
        # use the gpu provider.
        self.provider = provider
        if self.provider == "cuda":
            try:
                if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                    logger.warning(
                        "CUDA provider not available for ONNX. Falling back to CPU."
                    )
                    self.provider = "cpu"
            except ImportError:
                logger.warning("ONNX Runtime not installed. Falling back to CPU.")
                self.provider = "cpu"
        logger.info(f"Sherpa-Onnx-ASR: Using {self.provider} for inference")

        self.recognizer = self._create_recognizer()

    def _create_recognizer(self):
        if self.model_type == "sense_voice":
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=self.sense_voice,
                tokens=self.tokens,
                num_threads=self.num_threads,
                use_itn=self.use_itn,
                debug=self.debug,
                provider=self.provider,
            )
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        return recognizer

    def transcribe_np(self, audio: np.ndarray) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.SAMPLE_RATE, audio)
        self.recognizer.decode_streams([stream])
        return stream.result.text

    def transcribe_from_mic_with_vad(self, aggressiveness=0, frame_duration=30):
        """
        从麦克风录制音频并进行转录，先录制 3 秒，再使用 VAD 检测语音停止
        :param aggressiveness: VAD 检测的激进程度
        :param frame_duration: 音频帧持续时间（毫秒）
        :return: 转录后的文本
        """
        vad = webrtcvad.Vad(aggressiveness)
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=int(self.SAMPLE_RATE * frame_duration / 1000))

        print("开始录制...")
        frames = []
        # 之后使用 VAD 判断是否停止说话
        while True:
            data = stream.read(int(self.SAMPLE_RATE * frame_duration / 1000))
            is_speech = vad.is_speech(data, self.SAMPLE_RATE)
            if is_speech:
                frames.append(data)
                non_speech_frames_count = 0
            else:
                non_speech_frames_count += 1
                if non_speech_frames_count >= 35:
                    if frames:
                        break
                else:
                    frames.append(data)

        print("录制结束...")

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32767.0  # 将音频数据转换为 [-1, 1] 范围
        return self.transcribe_np(audio_data)



    