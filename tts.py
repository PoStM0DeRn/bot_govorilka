import logging
import re

from num2words import num2words

from config import TTS_SAMPLE_RATE

logger = logging.getLogger(__name__)

_model = None
_speakers = None

MAX_TEXT_LENGTH = 2000

_TAG_PATTERN = re.compile(r'\[emotion:\w+\]')
_MULTI_SPACE = re.compile(r' {2,}')
_MULTI_NEWLINE = re.compile(r'\n{2,}')

_TRANSTABLE = str.maketrans({
    'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф',
    'g': 'г', 'h': 'х', 'i': 'и', 'j': 'й', 'k': 'к', 'l': 'л',
    'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р',
    's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс',
    'y': 'и', 'z': 'з',
    'A': 'А', 'B': 'Б', 'C': 'К', 'D': 'Д', 'E': 'Е', 'F': 'Ф',
    'G': 'Г', 'H': 'Х', 'I': 'И', 'J': 'Й', 'K': 'К', 'L': 'Л',
    'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'Q': 'К', 'R': 'Р',
    'S': 'С', 'T': 'Т', 'U': 'У', 'V': 'В', 'W': 'В', 'X': 'Кс',
    'Y': 'И', 'Z': 'З',
})

_EN_WORD = re.compile(r'[A-Za-z]{2,}')
_EN_CK = re.compile(r'кк')
_EN_DOUBLE = re.compile(r'([бвгджзклмнпрстфхцчшщ])\1')
_DIGITS = re.compile(r'\d[\d\s.,:]*\d|\d')

AVAILABLE_SPEAKERS = {
    "kseniya": "Ксения (женский)",
    "baya": "Бая (женский)",
    "xenia": "Ксения (женский, другой)",
    "aidar": "Аидар (мужской)",
    "eugene": "Евгений (мужской)",
}


def _numbers_to_words(text: str) -> str:
    """Заменить все числа в тексте на русские слова."""
    def _replace(match):
        raw = match.group(0).replace(" ", "")
        try:
            if "." in raw:
                return num2words(float(raw), lang="ru")
            return num2words(int(raw), lang="ru")
        except Exception:
            return raw
    return _DIGITS.sub(_replace, text)


def _transliterate_english(text: str) -> str:
    """Транслитерировать английские слова в кириллицу."""
    def _replace(match):
        word = match.group(0)
        result = word.translate(_TRANSTABLE)
        result = _EN_CK.sub("к", result)
        result = result.replace("тх", "т")
        result = result.replace("пх", "ф")
        result = result.replace("шх", "ш")
        result = result.replace("чх", "ч")
        return result
    return _EN_WORD.sub(_replace, text)


def load():
    global _model, _speakers
    import torch
    logger.info("Загрузка TTS модели (torch.hub, sr=%s)...", TTS_SAMPLE_RATE)
    _model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v5_ru',
    )
    _speakers = _model.speakers
    if isinstance(_speakers, dict):
        speaker_names = list(_speakers.keys())
    else:
        speaker_names = list(_speakers)
    logger.info("TTS модель загружена. Доступные голоса: %s", speaker_names)


def get_speakers() -> list:
    """Возвращает список доступных голосов."""
    if _model is None:
        return list(AVAILABLE_SPEAKERS.keys())
    if isinstance(_speakers, dict):
        return list(_speakers.keys())
    return list(_speakers)


def _clean_text(text: str) -> str:
    """Очистить текст для совместимости с Silero TTS."""
    text = _TAG_PATTERN.sub("", text)
    text = _numbers_to_words(text)
    text = _transliterate_english(text)
    text = re.sub(r'[*#_`~\[\]{}()<>|\\/]', '', text)
    text = re.sub(r'[^\w\s.,!?;:\-—–…«»\"\'ЁёА-яа-яA-Za-z0-9]', '', text)
    text = _MULTI_SPACE.sub(' ', text)
    text = _MULTI_NEWLINE.sub('. ', text)
    return text.strip()


def synthesize(text: str, speaker: str = None) -> tuple:
    """Синтез речи. Возвращает (audio_float32, sample_rate)."""
    import numpy as np
    from config import TTS_SPEAKER

    if _model is None:
        raise RuntimeError("TTS модель не загружена. Вызовите load() сначала.")

    if not text or not text.strip():
        raise ValueError("Пустой текст для синтеза.")

    text = _clean_text(text)
    if not text:
        raise ValueError("Текст пуст после очистки.")

    if len(text) > MAX_TEXT_LENGTH:
        logger.warning("Текст обрезан с %d до %d символов.", len(text), MAX_TEXT_LENGTH)
        text = text[:MAX_TEXT_LENGTH]

    use_speaker = speaker or TTS_SPEAKER

    try:
        audio_tensor = _model.apply_tts(
            text=text,
            speaker=use_speaker,
            sample_rate=TTS_SAMPLE_RATE,
        )

        audio = audio_tensor.cpu().numpy().astype(np.float32)

        silence_len = int(TTS_SAMPLE_RATE * 0.4)
        silence = np.zeros(silence_len, dtype=np.float32)
        audio = np.concatenate([audio, silence])

        fade_len = int(TTS_SAMPLE_RATE * 0.2)
        audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len).astype(np.float32)

        logger.debug("TTS синтезировано: %d сэмплов, sr=%d, голос=%s", len(audio), TTS_SAMPLE_RATE, use_speaker)
        return audio, TTS_SAMPLE_RATE

    except Exception as e:
        logger.error("Ошибка TTS синтеза (голос=%s): %s", use_speaker, e, exc_info=True)
        raise
