import atexit
import logging
import logging.handlers
import os
import signal
import sys
import threading
from pathlib import Path

from audio import record_with_vad, play_audio, listen_with_wake_word
from stt import load as load_stt, transcribe_stream
from llm import ask_with_tools, load_history, switch_profile_history
from tts import load as load_tts, synthesize
from avatar import AvatarWindow
from emotions import extract_emotion
from config import validate_config, WAKE_WORD_ENABLED, FACE_RECOGNITION_ENABLED, MAX_INPUT_LENGTH
from metrics import metrics
import profiles
import face_engine

logger = logging.getLogger(__name__)

_avatar = None
_shutdown_event = threading.Event()
_registration_mode = False
_registration_event = threading.Event()


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "assistant.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    root.addHandler(console)
    root.addHandler(file_handler)


def _signal_handler(signum, frame):
    logger.info("Получен сигнал %d, завершаю работу...", signum)
    _shutdown_event.set()


def _cleanup():
    global _avatar
    logger.info("Очистка ресурсов...")
    face_engine.stop_camera()
    if _avatar is not None:
        try:
            _avatar.stop()
        except Exception as e:
            logger.error("Ошибка остановки аватара: %s", e)
    logger.info("Готово.")


def _on_face_detected(name: str):
    """Callback при обнаружении нового лица."""
    global _registration_mode
    if _registration_mode:
        return

    profile = profiles.switch_to(name)
    switch_profile_history()
    greeting = profile.get("greeting", f"Привет, {name}!")
    voice = profile.get("tts_voice", "kseniya")
    logger.info("Обнаружен: %s (голос: %s)", name, voice)
    print(f"\n  [Камера] Обнаружен: {name}")
    try:
        audio, sr = synthesize(greeting, speaker=voice)
        play_audio(audio, sr)
    except Exception as e:
        logger.error("Ошибка приветствия: %s", e)


def _speak(text: str, emotion: str = "neutral", voice: str = None):
    if _avatar:
        _avatar.set_emotion(emotion, duration=0.3)
    with metrics.timing("tts"):
        audio, sr = synthesize(text, speaker=voice)
    lip_cb = _avatar.play_audio_with_lipsync if _avatar else None
    play_audio(audio, sr, lip_sync_callback=lip_cb)


def _speak_in_thread(text: str, done_event: threading.Event, emotion: str = "neutral", voice: str = None):
    try:
        _speak(text, emotion, voice)
    except Exception as e:
        logger.error("Ошибка воспроизведения: %s", e)
    finally:
        done_event.set()


def _handle_registration(audio_data):
    """Обработка регистрации нового лица."""
    global _registration_mode

    print("  Распознаю...")
    user_text = ""
    for word, start, end in transcribe_stream(audio_data):
        user_text += word + " "
    user_text = user_text.strip()
    print(f"  Вы: {user_text}")

    if not user_text:
        print("  Не удалось распознать. Попробуйте снова.")
        _registration_mode = False
        _registration_event.set()
        return

    if _avatar:
        frame = _avatar.get_camera_frame()
        if frame is None:
            print("  Камера недоступна. Регистрация через камеру невозможна.")
            _speak("Камера недоступна. Попробуйте позже.")
            _registration_mode = False
            _registration_event.set()
            return

        encoding = face_engine.capture_face_for_registration(frame)
        if encoding is None:
            print("  Лицо не обнаружено. Посмотрите в камеру.")
            _speak("Я не вижу лицо. Посмотрите в камеру и попробуйте снова.")
            _registration_mode = False
            _registration_event.set()
            return

        name = user_text
        face_engine.save_face(name, encoding)
        profiles.create_profile(name)
        profiles.switch_to(name)
        switch_profile_history()

        print(f"  {name}, вы запомнены!")
        _speak(f"Привет, {name}! Теперь я буду знать тебя.")
        _registration_mode = False
        _registration_event.set()
    else:
        print("  Аватар недоступен. Регистрация невозможна.")
        _registration_mode = False
        _registration_event.set()


def main():
    global _avatar, _registration_mode

    setup_logging()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup)

    logger.info("=== Голосовой ассистент ===")

    errors = validate_config()
    if errors:
        for err in errors:
            logger.error("Ошибка конфигурации: %s", err)
        sys.exit(1)

    logger.info("Загрузка моделей...")

    try:
        load_stt()
    except Exception as e:
        logger.critical("Не удалось загрузить STT модель: %s", e)
        sys.exit(1)

    try:
        load_tts()
    except Exception as e:
        logger.critical("Не удалось загрузить TTS модель: %s", e)
        sys.exit(1)

    profiles.init()
    profiles.switch_to("Гость")
    load_history()

    if FACE_RECOGNITION_ENABLED:
        face_engine.load_known_faces()
        face_engine.start_camera()

    logger.info("Модели загружены.")

    logger.info("Запуск аватара...")
    try:
        _avatar = AvatarWindow()
        if FACE_RECOGNITION_ENABLED:
            _avatar.set_face_callback(_on_face_detected)
        _avatar.start()
        logger.info("Аватар запущен.")
    except Exception as e:
        logger.error("Не удалось запустить аватар: %s. Продолжаем без аватара.", e)
        _avatar = None

    current_profile = profiles.get_current()
    profile_name = current_profile.get("name", "Гость") if current_profile else "Гость"

    print("Введите текст или нажмите Enter для голосового ввода.")
    if WAKE_WORD_ENABLED:
        print("Режим wake word: скажите 'hey_jarvis' для активации.")
    else:
        print("Голосовой ввод: нажмите Enter → говорите → автоматический стоп по тишине.")
    if FACE_RECOGNITION_ENABLED:
        print("Скажите 'запомни меня' для регистрации нового лица.")
    print("Команды: 'выход' / 'quit' — выход из программы.")
    print(f"  Текущий профиль: {profile_name}")

    try:
        while not _shutdown_event.is_set():
            print()
            try:
                if WAKE_WORD_ENABLED:
                    audio = listen_with_wake_word()
                    if len(audio) < 1600:
                        continue
                    print("  Распознаю...")
                    user_text = ""
                    with metrics.timing("stt"):
                        for word, start, end in transcribe_stream(audio):
                            print(word, end=" ", flush=True)
                            user_text += word + " "
                    user_text = user_text.strip()
                    print()
                    if not user_text:
                        print("  Не удалось распознать речь.")
                        continue
                    print(f"  Вы (голос): {user_text}")
                else:
                    user_input = input("Вы: ").strip()
                    if not user_input:
                        audio = record_with_vad()
                        if len(audio) < 1600:
                            print("  Слишком коротко, попробуйте снова.")
                            continue
                        print("  Распознаю...")
                        user_text = ""
                        with metrics.timing("stt"):
                            for word, start, end in transcribe_stream(audio):
                                print(word, end=" ", flush=True)
                                user_text += word + " "
                        user_text = user_text.strip()
                        print()
                        if not user_text:
                            print("  Не удалось распознать речь.")
                            continue
                        print(f"  Вы (голос): {user_text}")
                    else:
                        user_text = user_input
                        if user_text.lower() in ("выход", "стоп", "quit", "exit"):
                            break

                if len(user_text) > MAX_INPUT_LENGTH:
                    print(f"  Ввод слишком длинный ({len(user_text)} > {MAX_INPUT_LENGTH}). Сократите.")
                    continue

                if "запомни меня" in user_text.lower() and FACE_RECOGNITION_ENABLED:
                    print("  Режим регистрации. Скажите своё имя...")
                    _speak("Смотрю в камеру. Скажите своё имя.")
                    _registration_mode = True
                    _registration_event.clear()

                    reg_audio = record_with_vad()
                    if len(reg_audio) < 1600:
                        print("  Слишком коротко. Попробуйте снова.")
                        _registration_mode = False
                        continue

                    _handle_registration(reg_audio)
                    _registration_event.wait(timeout=10)
                    continue

            except (KeyboardInterrupt, EOFError):
                print()
                break

            print("  Думаю...")
            print("  Ассистент: ", end="", flush=True)

            current_profile = profiles.get_current()
            voice = current_profile.get("tts_voice") if current_profile else None

            full_response = ""
            try:
                for token in ask_with_tools(user_text):
                    print(token, end="", flush=True)
                    full_response += token
            except ConnectionError as e:
                logger.error("Ошибка LLM: %s", e)
                print(f"\n  Ошибка: {e}")
                continue
            except Exception as e:
                logger.error("Неожиданная ошибка LLM: %s", e, exc_info=True)
                print(f"\n  Ошибка: {e}")
                continue

            clean_text, emotion = extract_emotion(full_response)
            logger.info("Эмоция ответа: %s", emotion)

            if _avatar:
                _avatar.set_emotion(emotion, duration=0.3)

            if not clean_text:
                print()
                continue

            speak_thread = None
            speak_event = threading.Event()

            buffer = clean_text
            while buffer:
                last_end = max(buffer.rfind("."), buffer.rfind("!"), buffer.rfind("?"))
                if last_end != -1:
                    chunk = buffer[:last_end + 1].strip()
                    buffer = buffer[last_end + 1:]
                    if chunk:
                        if speak_thread is not None:
                            speak_event.wait()
                        speak_event.clear()
                        speak_thread = threading.Thread(
                            target=_speak_in_thread,
                            args=(chunk, speak_event, emotion, voice),
                            daemon=True,
                        )
                        speak_thread.start()
                else:
                    break

            remaining = buffer.strip()
            if remaining:
                if speak_thread is not None:
                    speak_event.wait()
                try:
                    _speak(remaining, emotion, voice)
                except Exception as e:
                    logger.error("Ошибка финального воспроизведения: %s", e)
            elif speak_thread is not None:
                speak_event.wait()

            print()

    finally:
        logger.info("Завершение работы...")
        _cleanup()


if __name__ == "__main__":
    main()
