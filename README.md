# Голосовой ассистент «Кей»

Голосовой ассистент с Live2D-аватаром, работающий полностью локально.

## Возможности

- **STT:** faster-whisper (авто-определение GPU/CPU)
- **VAD:** Silero VAD — автоматическое определение начала/конца речи
- **LLM:** LM Studio (OpenAI-совместимый API)
- **TTS:** Silero TTS (русский голос «Ксения»)
- **Аватар:** Live2D с lip-sync

## Установка

```bash
# Клонировать репозиторий
git clone <url>
cd bot-govorilka

# Создать виртуальное окружение
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Установить зависимости
pip install -r requirements.txt
```

### Системные требования

- Python 3.10+
- Микрофон
- LM Studio (запущенный на `localhost:1234`)
- (Опционально) NVIDIA GPU для ускорения STT

## Настройка

### Через `.env` файл

Создайте файл `.env` в корне проекта:

```env
# Whisper
WHISPER_MODEL=small
WHISPER_LANGUAGE=ru
# WHISPER_DEVICE=cuda    # раскомментировать для GPU
# WHISPER_COMPUTE_TYPE=float16

# VAD
VAD_THRESHOLD=0.5
VAD_SILENCE_TIMEOUT_MS=600

# LM Studio
LM_STUDIO_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=

# TTS
TTS_SPEAKER=kseniya
TTS_SAMPLE_RATE=48000

# Аватар
AVATAR_MODEL_PATH=Resources/Kei/kei_vowels_pro/runtime/kei_vowels_pro.model3.json
AVATAR_WIDTH=500
AVATAR_HEIGHT=600
```

### Через переменные окружения

Все параметры конфигурации доступны как переменные окружения с теми же именами.

## Запуск

```bash
python main.py
```

### Использование

1. Запустите LM Studio и загрузите модель
2. Запустите `python main.py`
3. **Текстовый ввод:** просто напишите сообщение
4. **Голосовой ввод:** нажмите Enter → говорите → автоматический стоп по тишине
5. **Выход:** напишите `выход` или `quit`

## Архитектура

```
Микрофон → [VAD] → [faster-whisper] → Текст → [LM Studio] → Ответ → [Silero TTS] → Аудио → [Live2D]
```

| Модуль | Ответственность |
|--------|----------------|
| `main.py` | Оркестратор, главный цикл |
| `audio.py` | Запись/воспроизведение аудио, VAD |
| `stt.py` | Speech-to-Text (faster-whisper) |
| `tts.py` | Text-to-Speech (Silero TTS) |
| `llm.py` | Языковая модель (LM Studio) |
| `avatar.py` | Live2D аватар с lip-sync |
| `config.py` | Конфигурация (.env, env vars) |

## Тесты

```bash
pip install pytest
pytest
```

## Лицензия

MIT
