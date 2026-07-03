import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default=None, cast=None):
    raw = os.getenv(key)
    if raw is None:
        return default
    if cast is not None:
        try:
            return cast(raw)
        except (ValueError, TypeError):
            logger.warning("Невозможно преобразовать %s=%s в %s, используется значение по умолчанию: %s",
                         key, raw, cast.__name__, default)
            return default
    return raw


WHISPER_MODEL = _env("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE", "ru")
WHISPER_DEVICE = _env("WHISPER_DEVICE", None)
WHISPER_COMPUTE_TYPE = _env("WHISPER_COMPUTE_TYPE", None)

VAD_THRESHOLD = _env("VAD_THRESHOLD", 0.5, float)
VAD_SILENCE_TIMEOUT_MS = _env("VAD_SILENCE_TIMEOUT_MS", 600, int)
VAD_SPEECH_PAD_MS = _env("VAD_SPEECH_PAD_MS", 200, int)
VAD_MIN_SPEECH_MS = _env("VAD_MIN_SPEECH_MS", 100, int)

LM_STUDIO_URL = _env("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = _env("LM_STUDIO_MODEL", "")

TTS_SPEAKER = _env("TTS_SPEAKER", "kseniya")
TTS_SAMPLE_RATE = _env("TTS_SAMPLE_RATE", 48000, int)

SYSTEM_PROMPT = _env("SYSTEM_PROMPT",
    "Ты — голосовой ассистент по имени Кей. "
    "Ты дружелюбный, немного дерзкий, но полезный. "
    "Отвечай кратко (1-3 предложения), на том языке, на котором пишет собеседник. "
    "Используй разговорный стиль.\n\n"
    "В конце КАЖДОГО ответа добавляй тег эмоции на отдельной строке:\n"
    "[emotion:happy] — радость, восторг, согласие\n"
    "[emotion:sad] — грусть, сожаление\n"
    "[emotion:angry] — злость, раздражение\n"
    "[emotion:surprised] — удивление, неожиданность\n"
    "[emotion:thinking] — задумчивость, обдумывание\n"
    "[emotion:embarrassed] — смущение, неловкость\n"
    "[emotion:neutral] — спокойствие, нейтральность (по умолчанию)\n\n"
    "Пример:\nПривет! Конечно помогу тебе с этим!\n[emotion:happy]"
)
MAX_HISTORY = _env("MAX_HISTORY", 20, int)
HISTORY_FILE = _env("HISTORY_FILE", "history.json")

AVATAR_MODEL_PATH = _env("AVATAR_MODEL_PATH", "Resources/Kei/kei_vowels_pro/runtime/kei_vowels_pro.model3.json")
AVATAR_WIDTH = _env("AVATAR_WIDTH", 500, int)
AVATAR_HEIGHT = _env("AVATAR_HEIGHT", 600, int)
AVATAR_LIP_SYNC_MULT = _env("AVATAR_LIP_SYNC_MULT", 3.0, float)
AVATAR_BACKGROUND = _env("AVATAR_BACKGROUND", None)

MAX_INPUT_LENGTH = _env("MAX_INPUT_LENGTH", 500, int)
MAX_AUDIO_SECONDS = _env("MAX_AUDIO_SECONDS", 60, int)

WEATHER_DEFAULT_CITY = _env("WEATHER_DEFAULT_CITY", "Москва")
SEARCH_MAX_RESULTS = _env("SEARCH_MAX_RESULTS", 3, int)

WAKE_WORD_ENABLED = _env("WAKE_WORD_ENABLED", False, lambda x: x.lower() in ("true", "1", "yes"))
WAKE_WORD_MODEL = _env("WAKE_WORD_MODEL", "hey_jarvis")
WAKE_WORD_THRESHOLD = _env("WAKE_WORD_THRESHOLD", 0.5, float)

FACE_RECOGNITION_ENABLED = _env("FACE_RECOGNITION_ENABLED", False, lambda x: x.lower() in ("true", "1", "yes"))
FACE_RECOGNITION_TOLERANCE = _env("FACE_RECOGNITION_TOLERANCE", 0.6, float)
FACE_RECOGNITION_MODEL = _env("FACE_RECOGNITION_MODEL", "hog")
CAMERA_INDEX = _env("CAMERA_INDEX", 0, int)

PYTHON_SANDBOX_TIMEOUT = _env("PYTHON_SANDBOX_TIMEOUT", 10, int)
PYTHON_SANDBOX_MAX_OUTPUT = _env("PYTHON_SANDBOX_MAX_OUTPUT", 2000, int)


def validate_config() -> list[str]:
    errors = []

    if WHISPER_MODEL not in ("tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"):
        errors.append(f"Неизвестная модель Whisper: {WHISPER_MODEL}")

    avatar_path = Path(AVATAR_MODEL_PATH)
    if not avatar_path.exists():
        errors.append(f"Модель аватара не найдена: {avatar_path}")

    history_path = Path(HISTORY_FILE)
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        errors.append(f"Невозможно создать директорию для истории: {e}")

    import urllib.request
    try:
        urllib.request.urlopen(f"{LM_STUDIO_URL}/models", timeout=3)
    except Exception:
        errors.append(f"LM Studio недоступен по адресу {LM_STUDIO_URL}")

    return errors
