import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILES_DIR = Path("faces/profiles")

_current_profile: dict | None = None
_profile_name: str | None = None

DEFAULT_PROFILE = {
    "name": "Гость",
    "tts_voice": "kseniya",
    "system_prompt": (
        "Ты — голосовой ассистент по имени Кей. "
        "Ты дружелюбный, немного дерзкий, но полезный. "
        "Отвечай кратко (1-3 предложения), на том языке, на котором пишет собеседник. "
        "Используй разговорный стиль."
    ),
    "greeting": "Привет! Я Кей, твой голосовой помощник.",
}


def init():
    """Создать директорию профилей если нет."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def get_profile_path(name: str) -> Path:
    return PROFILES_DIR / f"{name}.json"


def load_profile(name: str) -> dict:
    """Загрузить профиль по имени."""
    path = get_profile_path(name)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            logger.info("Профиль загружен: %s", name)
            return profile
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Ошибка загрузки профиля %s: %s", name, e)

    profile = dict(DEFAULT_PROFILE)
    profile["name"] = name
    save_profile(name, profile)
    return profile


def save_profile(name: str, data: dict):
    """Сохранить профиль."""
    path = get_profile_path(name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Профиль сохранён: %s", name)
    except IOError as e:
        logger.error("Ошибка сохранения профиля %s: %s", name, e)


def create_profile(name: str, tts_voice: str = "kseniya", system_prompt: str = None, greeting: str = None) -> dict:
    """Создать новый профиль."""
    profile = dict(DEFAULT_PROFILE)
    profile["name"] = name
    profile["tts_voice"] = tts_voice
    if system_prompt:
        profile["system_prompt"] = system_prompt
    if greeting:
        profile["greeting"] = greeting
    save_profile(name, profile)
    return profile


def delete_profile(name: str) -> bool:
    """Удалить профиль."""
    path = get_profile_path(name)
    if path.exists():
        path.unlink()
        logger.info("Профиль удалён: %s", name)
        return True
    return False


def list_profiles() -> list[str]:
    """В список имён всех профилей."""
    if not PROFILES_DIR.exists():
        return []
    return [p.stem for p in PROFILES_DIR.glob("*.json")]


def switch_to(name: str) -> dict:
    """Переключиться на профиль. Возвращает профиль."""
    global _current_profile, _profile_name
    _profile_name = name
    _current_profile = load_profile(name)
    logger.info("Переключение на профиль: %s", name)
    return _current_profile


def get_current() -> dict | None:
    """Текущий активный профиль."""
    return _current_profile


def get_current_name() -> str | None:
    """Имя текущего профиля."""
    return _profile_name


def get_history_path(name: str = None) -> str:
    """Путь к файлу истории для профиля."""
    name = name or _profile_name or "default"
    return f"history_{name}.json"
