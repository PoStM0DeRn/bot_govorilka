import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

EMOTION_PATTERN = re.compile(r'\[emotion:(\w+)\]')

VALID_EMOTIONS = ("happy", "sad", "angry", "surprised", "thinking", "neutral", "embarrassed")


@dataclass
class EmotionState:
    name: str = "neutral"
    params: dict = field(default_factory=dict)
    start_time: float = 0.0
    duration: float = 0.5


EXPRESSION_PRESETS = {
    "happy": {
        "ParamMouthForm": 0.8,
        "ParamEyeLSmile": 0.8,
        "ParamEyeRSmile": 0.8,
        "ParamCheek": 0.4,
        "ParamBrowLY": 0.2,
        "ParamBrowRY": 0.2,
    },
    "sad": {
        "ParamMouthForm": -0.6,
        "ParamBrowLAngle": -0.4,
        "ParamBrowRAngle": -0.4,
        "ParamBrowLY": -0.3,
        "ParamBrowRY": -0.3,
        "ParamEyeLOpen": 0.6,
        "ParamEyeROpen": 0.6,
        "ParamAngleY": -8,
    },
    "angry": {
        "ParamMouthForm": -0.8,
        "ParamBrowLAngle": -0.8,
        "ParamBrowRAngle": -0.8,
        "ParamBrowLForm": -0.8,
        "ParamBrowRForm": -0.8,
        "ParamBrowLY": -0.4,
        "ParamBrowRY": -0.8,
        "ParamAngleZ": 3,
    },
    "surprised": {
        "ParamEyeLOpen": 1.2,
        "ParamEyeROpen": 1.2,
        "ParamBrowLY": 0.7,
        "ParamBrowRY": 0.7,
        "ParamMouthOpenY": 0.5,
        "ParamMouthForm": 0.2,
    },
    "thinking": {
        "ParamEyeBallX": 0.6,
        "ParamEyeBallY": 0.2,
        "ParamAngleZ": -5,
        "ParamAngleX": 8,
        "ParamBrowLY": 0.25,
        "ParamBrowRY": -0.15,
        "ParamMouthForm": -0.15,
    },
    "neutral": {
        "ParamMouthForm": 0.0,
        "ParamEyeLSmile": 0.0,
        "ParamEyeRSmile": 0.0,
        "ParamCheek": 0.0,
        "ParamBrowLY": 0.0,
        "ParamBrowRY": 0.0,
        "ParamBrowLAngle": 0.0,
        "ParamBrowRAngle": 0.0,
        "ParamBrowLForm": 0.0,
        "ParamBrowRForm": 0.0,
        "ParamEyeLOpen": 1.0,
        "ParamEyeROpen": 1.0,
        "ParamEyeBallX": 0.0,
        "ParamEyeBallY": 0.0,
        "ParamAngleX": 0.0,
        "ParamAngleY": 0.0,
        "ParamAngleZ": 0.0,
        "ParamMouthOpenY": 0.0,
    },
    "embarrassed": {
        "ParamMouthForm": 0.3,
        "ParamCheek": 0.6,
        "ParamAngleY": -6,
        "ParamAngleZ": 4,
        "ParamEyeLOpen": 0.7,
        "ParamEyeROpen": 0.7,
        "ParamEyeBallY": -0.3,
    },
}


def extract_emotion(text: str) -> tuple[str, str]:
    """Извлечь тег эмоции из текста. Возвращает (чистый_текст, эмоция)."""
    match = EMOTION_PATTERN.search(text)
    if match:
        emotion = match.group(1).lower()
        if emotion not in VALID_EMOTIONS:
            logger.warning("Неизвестная эмоция '%s', используется neutral.", emotion)
            emotion = "neutral"
        clean_text = EMOTION_PATTERN.sub("", text).strip()
        return clean_text, emotion
    return text.strip(), "neutral"


def get_emotion_params(emotion: str) -> dict:
    """Получить параметры Live2D для эмоции."""
    return EXPRESSION_PRESETS.get(emotion, EXPRESSION_PRESETS["neutral"]).copy()


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t
