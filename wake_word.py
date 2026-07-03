import logging
import numpy as np

logger = logging.getLogger(__name__)

WAKE_WORD_FRAME_SIZE = 1280

_model = None
_wake_word_name = None
_is_enabled = False
_threshold = 0.5


def load(wake_word_model: str = "hey_jarvis", threshold: float = 0.5):
    """Загрузить модель wake word. Возвращает True если успешно."""
    global _model, _wake_word_name, _is_enabled, _threshold

    try:
        import openwakeword
        from openwakeword.model import Model

        logger.info("Загрузка wake word модели: %s", wake_word_model)
        openwakeword.utils.download_models()

        _model = Model(
            wakeword_models=[wake_word_model],
            vad_threshold=threshold,
        )
        _wake_word_name = wake_word_model
        _threshold = threshold
        _is_enabled = True
        logger.info("Wake word модель загружена: %s (threshold=%.2f)", wake_word_model, threshold)
        return True

    except ImportError:
        logger.warning("openwakeword не установлен. Wake word отключён.")
        _is_enabled = False
        return False
    except Exception as e:
        logger.error("Ошибка загрузки wake word: %s", e)
        _is_enabled = False
        return False


def is_enabled() -> bool:
    return _is_enabled


def detect(audio_frame: np.ndarray) -> bool:
    """Проверить фрейм аудио на наличие wake word.
    
    audio_frame: numpy array, ровно 1280 сэмплов (80мс при 16kHz), float32.
    Возвращает True если wake word обнаружен.
    """
    if not _is_enabled or _model is None:
        return False

    if len(audio_frame) < WAKE_WORD_FRAME_SIZE:
        return False

    try:
        prediction = _model.predict(audio_frame[:WAKE_WORD_FRAME_SIZE])
        score = prediction.get(_wake_word_name, 0)

        if score > _threshold:
            logger.debug("Wake word обнаружен (score=%.2f)", score)
            return True

        return False

    except Exception as e:
        logger.error("Ошибка wake word детекции: %s", e)
        return False


def reset():
    """Сбросить состояние модели."""
    if _model is not None:
        _model.reset()
