import logging

import numpy as np
import torch
from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    VAD_THRESHOLD, VAD_SILENCE_TIMEOUT_MS, VAD_SPEECH_PAD_MS,
)

logger = logging.getLogger(__name__)

_model = None


def load():
    global _model
    device = WHISPER_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = WHISPER_COMPUTE_TYPE or ("float16" if device == "cuda" else "int8")
    logger.info("Whisper: модель=%s, устройство=%s, точность=%s", WHISPER_MODEL, device, compute_type)
    _model = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
    )
    logger.info("Whisper модель загружена.")


def transcribe(audio: np.ndarray) -> str:
    if _model is None:
        raise RuntimeError("Модель не загружена. Вызовите load() сначала.")
    if len(audio) < 1600:
        return ""
    segments, info = _model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=5,
        vad_filter=True,
        vad_parameters={
            "threshold": VAD_THRESHOLD,
            "min_silence_duration_ms": VAD_SILENCE_TIMEOUT_MS,
            "speech_pad_ms": VAD_SPEECH_PAD_MS,
        },
    )
    text = " ".join(s.text.strip() for s in segments)
    logger.debug("STT: язык=%s, вероятность=%.2f", info.language, info.language_probability)
    return text


def transcribe_stream(audio: np.ndarray):
    if _model is None:
        raise RuntimeError("Модель не загружена. Вызовите load() сначала.")
    if len(audio) < 1600:
        return
    segments, info = _model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=5,
        vad_filter=True,
        vad_parameters={
            "threshold": VAD_THRESHOLD,
            "min_silence_duration_ms": VAD_SILENCE_TIMEOUT_MS,
            "speech_pad_ms": VAD_SPEECH_PAD_MS,
        },
        word_timestamps=True,
    )
    logger.debug("STT stream: язык=%s", info.language)
    for segment in segments:
        if segment.words:
            for word in segment.words:
                yield word.word, word.start, word.end
        else:
            yield segment.text.strip(), segment.start, segment.end
