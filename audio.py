import logging
import time
from collections import deque

import numpy as np
import queue
import sounddevice as sd
import threading
import torch

from config import VAD_THRESHOLD, VAD_SILENCE_TIMEOUT_MS, VAD_SPEECH_PAD_MS, VAD_MIN_SPEECH_MS
from config import WAKE_WORD_ENABLED, WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD, MAX_AUDIO_SECONDS

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_FRAME_SIZE = 512
WAKE_FRAME_SIZE = 1280

MAX_AUDIO_CHUNKS = MAX_AUDIO_SECONDS * SAMPLE_RATE // VAD_FRAME_SIZE

_vad_model = None


def _load_vad():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad
        logger.info("Загрузка Silero VAD...")
        _vad_model = load_silero_vad()
        logger.info("Silero VAD загружен.")


def _open_stream(retries=3, **kwargs):
    for attempt in range(retries):
        try:
            stream = sd.InputStream(**kwargs)
            stream.start()
            return stream
        except sd.PortAudioError as e:
            logger.warning("Ошибка аудиоустройства (попытка %d/%d): %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                devices = sd.query_devices()
                logger.error("Доступные устройства:\n%s", devices)
                raise RuntimeError(
                    f"Не удалось открыть микрофон после {retries} попыток. "
                    f"Проверьте подключение микрофона."
                ) from e


def record_push_to_talk() -> np.ndarray:
    """Запись по нажатию Enter (старый режим, оставлен как fallback)."""
    audio_queue = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if not stop_event.is_set():
            audio_queue.put(indata.copy())

    logger.info("Push-to-talk: начинаю запись...")
    print("  Записываю... [Enter] - остановить")

    try:
        with _open_stream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=audio_callback, blocksize=int(SAMPLE_RATE * 0.1),
        ):
            input()
            stop_event.set()
    except RuntimeError:
        raise
    except Exception as e:
        logger.error("Ошибка записи: %s", e, exc_info=True)
        return np.array([], dtype="float32")

    chunks = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get())
    if not chunks:
        return np.array([], dtype="float32")
    return np.concatenate(chunks, axis=0).flatten()


def record_with_vad() -> np.ndarray:
    """Запись с VAD. Начало по Enter, автоматический стоп по тишине."""
    _load_vad()

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if not stop_event.is_set():
            audio_queue.put(indata.copy())

    logger.info("VAD запись: начинаю...")
    print("  Начинайте говорить... [Enter] - остановить вручную")

    speech_chunks = []
    speech_active = False
    silence_start = None
    speech_start = None
    pad_chunks = deque()

    silence_timeout_sec = VAD_SILENCE_TIMEOUT_MS / 1000.0
    pad_samples_needed = int(SAMPLE_RATE * (VAD_SPEECH_PAD_MS / 1000.0))
    pad_max_chunks = pad_samples_needed // VAD_FRAME_SIZE + 2

    try:
        with _open_stream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=audio_callback, blocksize=VAD_FRAME_SIZE,
        ):
            input()  # ждём Enter чтобы начать

            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                chunk_flat = chunk.flatten()

                if len(chunk_flat) < VAD_FRAME_SIZE:
                    continue

                audio_tensor = torch.from_numpy(chunk_flat[:VAD_FRAME_SIZE])
                prob = _vad_model(audio_tensor, SAMPLE_RATE).item()

                if prob > VAD_THRESHOLD:
                    if not speech_active:
                        speech_active = True
                        speech_start = time.monotonic()
                        silence_start = None
                        speech_chunks.extend(list(pad_chunks)[-pad_samples_needed // VAD_FRAME_SIZE:])
                        pad_chunks.clear()
                    speech_chunks.append(chunk_flat)
                    silence_start = None

                    if len(speech_chunks) > MAX_AUDIO_CHUNKS:
                        logger.warning("Превышен максимальный размер аудио (%d сек).", MAX_AUDIO_SECONDS)
                        break
                else:
                    if speech_active:
                        speech_chunks.append(chunk_flat)
                        if silence_start is None:
                            silence_start = time.monotonic()
                        elif (time.monotonic() - silence_start) >= silence_timeout_sec:
                            speech_duration_ms = (time.monotonic() - speech_start) * 1000
                            if speech_duration_ms < VAD_MIN_SPEECH_MS:
                                logger.debug("Короткая речь (%.0fмс), сброс.", speech_duration_ms)
                                speech_chunks.clear()
                                speech_active = False
                                speech_start = None
                            else:
                                break
                    else:
                        pad_chunks.append(chunk_flat)
                        if len(pad_chunks) > pad_max_chunks:
                            pad_chunks.popleft()

    except RuntimeError:
        raise
    except Exception as e:
        logger.error("Ошибка VAD записи: %s", e, exc_info=True)
        return np.array([], dtype="float32")

    if not speech_chunks:
        return np.array([], dtype="float32")

    audio = np.concatenate(speech_chunks)

    trim_samples = int(SAMPLE_RATE * 0.3)
    if len(audio) > trim_samples:
        audio = audio[:-trim_samples]

    logger.info("VAD запись завершена: %.1f сек.", len(audio) / SAMPLE_RATE)
    return audio


def play_audio(audio: np.ndarray, sample_rate: int, lip_sync_callback=None):
    """Воспроизведение аудио с опциональным lip-sync callback."""
    if lip_sync_callback:
        lip_sync_callback(audio, sample_rate)
    try:
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    except sd.PortAudioError as e:
        logger.error("Ошибка воспроизведения: %s", e)
    except Exception as e:
        logger.error("Непредвиденная ошибка воспроизведения: %s", e, exc_info=True)


def listen_with_wake_word() -> np.ndarray:
    """Непрерывное прослушивание. При обнаружении wake word → запись с VAD."""
    import wake_word

    if not wake_word.is_enabled():
        wake_word.load(WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD)

    if not wake_word.is_enabled():
        logger.info("Wake word отключён, используем record_with_vad().")
        return record_with_vad()

    _load_vad()

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if not stop_event.is_set():
            audio_queue.put(indata.copy())

    logger.info("Ожидание wake word '%s'...", WAKE_WORD_MODEL)
    print(f"  Скажите '{WAKE_WORD_MODEL}' для активации...")

    wake_buffer = np.array([], dtype=np.float32)

    try:
        with _open_stream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=audio_callback, blocksize=WAKE_FRAME_SIZE,
        ):
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                chunk_flat = chunk.flatten()

                wake_buffer = np.concatenate([wake_buffer, chunk_flat])
                while len(wake_buffer) >= WAKE_FRAME_SIZE:
                    frame = wake_buffer[:WAKE_FRAME_SIZE]
                    wake_buffer = wake_buffer[WAKE_FRAME_SIZE:]

                    if wake_word.detect(frame):
                        logger.info("Wake word обнаружен!")
                        print("  Слушаю...")
                        wake_word.reset()
                        return _record_speech_after_wake(audio_queue, stop_event)

    except RuntimeError:
        raise
    except Exception as e:
        logger.error("Ошибка wake word прослушивания: %s", e, exc_info=True)

    return np.array([], dtype="float32")


def _record_speech_after_wake(audio_queue: queue.Queue, stop_event: threading.Event) -> np.ndarray:
    """Записать речь после обнаружения wake word."""
    speech_chunks = []
    speech_active = False
    silence_start = None
    speech_start = None

    silence_timeout_sec = VAD_SILENCE_TIMEOUT_MS / 1000.0

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        chunk_flat = chunk.flatten()
        if len(chunk_flat) < VAD_FRAME_SIZE:
            continue

        audio_tensor = torch.from_numpy(chunk_flat[:VAD_FRAME_SIZE])
        prob = _vad_model(audio_tensor, SAMPLE_RATE).item()

        if prob > VAD_THRESHOLD:
            if not speech_active:
                speech_active = True
                speech_start = time.monotonic()
                silence_start = None
            speech_chunks.append(chunk_flat)
            silence_start = None

            if len(speech_chunks) > MAX_AUDIO_CHUNKS:
                break
        else:
            if speech_active:
                speech_chunks.append(chunk_flat)
                if silence_start is None:
                    silence_start = time.monotonic()
                elif (time.monotonic() - silence_start) >= silence_timeout_sec:
                    speech_duration_ms = (time.monotonic() - speech_start) * 1000
                    if speech_duration_ms < VAD_MIN_SPEECH_MS:
                        speech_chunks.clear()
                        speech_active = False
                        speech_start = None
                    else:
                        break

    if not speech_chunks:
        return np.array([], dtype="float32")

    audio = np.concatenate(speech_chunks)
    trim_samples = int(SAMPLE_RATE * 0.3)
    if len(audio) > trim_samples:
        audio = audio[:-trim_samples]

    logger.info("Wake word запись завершена: %.1f сек.", len(audio) / SAMPLE_RATE)
    return audio
