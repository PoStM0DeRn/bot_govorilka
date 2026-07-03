import logging
import os
from pathlib import Path

import numpy as np

from config import (
    FACE_RECOGNITION_ENABLED,
    FACE_RECOGNITION_TOLERANCE,
    FACE_RECOGNITION_MODEL,
)

logger = logging.getLogger(__name__)

KNOWN_FACES_DIR = Path("faces/known_faces")

_face_recognition = None
_video = None
_known_encodings = []
_known_names = []


def _load_lib():
    global _face_recognition
    if _face_recognition is None:
        try:
            import face_recognition
            _face_recognition = face_recognition
            logger.info("face_recognition загружен (модель: %s).", FACE_RECOGNITION_MODEL)
        except ImportError:
            logger.warning("face_recognition не установлен. Распознавание лиц отключено.")
            return False
    return True


def load_known_faces():
    """Загрузить все сохранённые encodings из faces/known_faces/."""
    global _known_encodings, _known_names

    if not _load_lib():
        return

    _known_encodings = []
    _known_names = []

    if not KNOWN_FACES_DIR.exists():
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
        return

    for path in KNOWN_FACES_DIR.glob("*.npy"):
        name = path.stem
        try:
            encoding = np.load(path)
            _known_encodings.append(encoding)
            _known_names.append(name)
            logger.info("Загружен encoding: %s", name)
        except Exception as e:
            logger.warning("Ошибка загрузки encoding %s: %s", name, e)

    logger.info("Всего загружено %d лиц.", len(_known_names))


def save_face(name: str, encoding: np.ndarray) -> bool:
    """Сохранить encoding лица в базу."""
    if not _load_lib():
        return False

    try:
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
        path = KNOWN_FACES_DIR / f"{name}.npy"
        np.save(path, encoding)
        _known_encodings.append(encoding)
        _known_names.append(name)
        logger.info("Лицо сохранено: %s", name)
        return True
    except Exception as e:
        logger.error("Ошибка сохранения лица %s: %s", name, e)
        return False


def remove_face(name: str) -> bool:
    """Удалить лицо из базы."""
    path = KNOWN_FACES_DIR / f"{name}.npy"
    if path.exists():
        path.unlink()
        load_known_faces()
        logger.info("Лицо удалено: %s", name)
        return True
    return False


def get_known_names() -> list[str]:
    """Возвращает список имён всех сохранённых лиц."""
    return list(_known_names)


def start_camera(index: int = 0):
    """Открыть камеру."""
    global _video
    if not FACE_RECOGNITION_ENABLED:
        return False

    try:
        import cv2
        _video = cv2.VideoCapture(index)
        if not _video.isOpened():
            logger.warning("Камера %d не найдена.", index)
            return False
        logger.info("Камера %d открыта.", index)
        return True
    except ImportError:
        logger.warning("opencv-python не установлен. Камера отключена.")
        return False


def stop_camera():
    """Закрыть камеру."""
    global _video
    if _video is not None:
        _video.release()
        _video = None
        logger.info("Камера закрыта.")


def read_frame():
    """Прочитать кадр с камеры. Возвращает (numpy_array, None) или (None, None)."""
    if _video is None:
        return None
    ret, frame = _video.read()
    if ret:
        return frame
    return None


def recognize(frame) -> tuple[list, list]:
    """
    Распознать лица на кадре.
    Возвращает (face_locations, face_names).
    face_locations — список (top, right, bottom, left).
    face_names — список имён или "Неизвестный".
    """
    if not _load_lib():
        return [], []

    if frame is None:
        return [], []

    rgb_frame = frame[:, :, ::-1]

    try:
        face_locations = _face_recognition.face_locations(
            rgb_frame, model=FACE_RECOGNITION_MODEL
        )
    except Exception as e:
        logger.debug("Ошибка детекции лиц: %s", e)
        return [], []

    if not face_locations:
        return [], []

    try:
        encodings = _face_recognition.face_encodings(rgb_frame, face_locations)
    except Exception as e:
        logger.debug("Ошибка encoding: %s", e)
        return face_locations, ["Неизвестный"] * len(face_locations)

    names = []
    for encoding in encodings:
        name = _match_face(encoding)
        names.append(name)

    return face_locations, names


def _match_face(encoding) -> str:
    """Сопоставить encoding с базой. Возвращает имя или 'Неизвестный'."""
    if not _known_encodings:
        return "Неизвестный"

    distances = _face_recognition.face_distance(_known_encodings, encoding)
    best_idx = int(np.argmin(distances))
    best_distance = distances[best_idx]

    if best_distance <= FACE_RECOGNITION_TOLERANCE:
        name = _known_names[best_idx]
        logger.debug("Лицо распознано: %s (расстояние: %.3f)", name, best_distance)
        return name

    logger.debug("Лицо не распознано (лучшее расстояние: %.3f)", best_distance)
    return "Неизвестный"


def capture_face_for_registration(frame) -> np.ndarray | None:
    """Захватить encoding лица с кадра для регистрации. Возвращает encoding или None."""
    if not _load_lib():
        return None

    if frame is None:
        return None

    rgb_frame = frame[:, :, ::-1]

    try:
        locations = _face_recognition.face_locations(rgb_frame, model=FACE_RECOGNITION_MODEL)
        if not locations:
            return None
        encodings = _face_recognition.face_encodings(rgb_frame, locations)
        if not encodings:
            return None
        return encodings[0]
    except Exception as e:
        logger.error("Ошибка захвата лица: %s", e)
        return None
