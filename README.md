# Голосовой ассистент «Кей»

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-78-brightgreen)

Голосовой ассистент с Live2D-аватаром, работающий **полностью локально** без облачных API.

## Возможности

### Голосовой ввод

- **STT:** faster-whisper с авто-определением GPU/CPU
- **VAD:** Silero VAD — автоматическое определение начала/конца речи
- **Wake Word:** опциональный режим hands-free активации (`hey_jarvis`, `alexa`)

### Голосовой вывод

- **TTS:** Silero TTS — 5 голосов: kseniya, baya, xenia, aidar, eugene
- Конвертация чисел в русские слова: `42` → «сорок два»
- Транслитерация английского: `Docker` → «Докер»

### Интерфейс

- **Live2D аватар** с lip-sync (60 fps во время речи)
- **Система эмоций:** happy, sad, angry, surprised, thinking, embarrassed, neutral
- **Камерный ввод** для распознавания лиц

### Инструменты (8 шт.)

| Инструмент | Описание |
|------------|----------|
| `get_time` | Текущая дата и время |
| `get_weather` | Погода через wttr.in |
| `set_timer` | Таймер с уведомлением |
| `calculate` | Математика (AST-безопасный eval) |
| `search_web` | Поиск DuckDuckGo |
| `system_info` | Информация о системе |
| `open_app` | Запуск приложений Windows |
| `run_python` | Python в изолированной песочнице |

### Профили

- Изолированные профили для каждого пользователя
- Face recognition для автоматического переключения
- Своя история, голос и системный промпт

## Установка

### Через pip

```bash
# Клонировать репозиторий
git clone https://github.com/your-username/bot-govorilka.git
cd bot-govorilka

# Создать виртуальное окружение
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Установить зависимости
pip install -r requirements.txt
```

### Через Docker

```bash
git clone https://github.com/your-username/bot-govorilka.git
cd bot-govorilka
docker compose up -d
```

### Системные требования

- Python 3.10+
- Микрофон
- LM Studio (запущенный на `localhost:1234`)
- (Опционально) NVIDIA GPU для ускорения STT
- (Опционально) Камера для face recognition

## Настройка

### Быстрый старт

Скопируйте `.env.example` в `.env` и настройте:

```bash
cp .env.example .env
```

Минимальная конфигурация:

```env
WHISPER_MODEL=small
LM_STUDIO_URL=http://localhost:1234/v1
TTS_SPEAKER=kseniya
```

### Полный список настроек

```env
# === Whisper (STT) ===
WHISPER_MODEL=small            # tiny/base/small/medium/large-v3/turbo
WHISPER_LANGUAGE=ru            # None = авто-определение языка
WHISPER_DEVICE=cuda            # cuda/cpu, None = auto
WHISPER_COMPUTE_TYPE=float16   # float16 (GPU) / int8 (CPU)

# === VAD (Voice Activity Detection) ===
VAD_THRESHOLD=0.5              # Порог детекции речи (0.0-1.0)
VAD_SILENCE_TIMEOUT_MS=600     # Тишина дольше (мс) → конец фразы
VAD_SPEECH_PAD_MS=200          # Паддинг вокруг речи (мс)
VAD_MIN_SPEECH_MS=100          # Минимальная длительность речи (мс)

# === Wake Word ===
WAKE_WORD_ENABLED=false        # Включить режим wake word
WAKE_WORD_MODEL=hey_jarvis     # hey_jarvis, alexa, и т.д.
WAKE_WORD_THRESHOLD=0.5        # Порог детекции

# === LM Studio (LLM) ===
LM_STUDIO_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=              # Имя модели (пусто = default)

# === Silero TTS ===
TTS_SPEAKER=kseniya           # kseniya, baya, xenia, aidar, eugene
TTS_SAMPLE_RATE=48000

# === История ===
MAX_HISTORY=20
HISTORY_FILE=history.json

# === Аватар ===
AVATAR_MODEL_PATH=Resources/Kei/kei_vowels_pro/runtime/kei_vowels_pro.model3.json
AVATAR_WIDTH=500
AVATAR_HEIGHT=600
AVATAR_LIP_SYNC_MULT=3.0

# === Face Recognition ===
FACE_RECOGNITION_ENABLED=false
FACE_RECOGNITION_TOLERANCE=0.6
CAMERA_INDEX=0

# === Прочее ===
MAX_INPUT_LENGTH=500
MAX_AUDIO_SECONDS=60
PYTHON_SANDBOX_TIMEOUT=10
```

## Запуск

```bash
python main.py
```

### Использование

| Режим | Как использовать |
|-------|------------------|
| Текстовый ввод | Просто напишите сообщение |
| Голосовой ввод | `Enter` → говорите → автостоп по тишине |
| Wake word | Скажите «hey_jarvis» (если включено) |
| Регистрация лица | Скажите «запомни меня» (если face recognition включён) |
| Выход | Напишите `выход` или `quit` |

### Инструменты LLM

Ассистент автоматически вызывает инструменты по необходимости:

```
Пользователь: Который час?
Ассистент: [вызывает get_time] Сейчас 15:42, среда, 3 июля.

Пользователь: Посчитай 2 + 2 * 3
Ассистент: [вызывает calculate] 2 + 2 * 3 = 8

Пользователь: Открой блокнот
Ассистент: [вызывает open_app] Блокнот запущен.

Пользователь: Напиши код сортировки пузырьком
Ассистент: [вызывает run_python] def bubble_sort(arr): ...
```

## Архитектура

```
Микрофон → [VAD] → [STT] → [LLM + Tools] → [TTS] → [Live2D Avatar]
```

| Модуль | Ответственность |
|--------|----------------|
| `main.py` | Оркестратор, главный цикл |
| `audio.py` | Запись/воспроизведение аудио, VAD |
| `stt.py` | Speech-to-Text (faster-whisper) |
| `tts.py` | Text-to-Speech (Silero TTS) |
| `llm.py` | LM Studio (OpenAI API), tool calling |
| `tools.py` | 8 инструментов для LLM |
| `avatar.py` | Live2D аватар с lip-sync |
| `emotions.py` | Система эмоций (lerp-интерполяция) |
| `face_engine.py` | Face recognition (face_recognition) |
| `profiles.py` | Профили пользователей |
| `config.py` | Конфигурация (.env, env vars) |
| `metrics.py` | Тайминг этапов пайплайна |
| `wake_word.py` | Wake word detection (openwakeword) |

## Тесты

```bash
# Запуск всех тестов
python -m pytest tests/ -v

# Запуск конкретного класса
python -m pytest tests/test_core.py::TestTools -v
```

### Покрытие (78 тестов)

| Модуль | Тесты |
|--------|-------|
| Конфигурация | Значения по умолчанию, override через env |
| LLM | История, circuit breaker, build messages |
| TTS | Синтез, очистка текста, числа в слова, транслитерация |
| Инструменты | 8 инструментов + Python sandbox (включая security) |
| Профили | CRUD, переключение, history path |
| Face engine | Загрузка, распознавание, регистрация |
| Эмоции | Извлечение тегов, параметры, lerp |
| Метрики | Тайминг, reset |

## Docker

### Сборка и запуск

```bash
docker compose up -d
```

### Остановка

```bash
docker compose down
```

### Логи

```bash
docker compose logs -f govorilka
```

### Сборка образа вручную

```bash
docker build -t govorilka .
docker run -it --rm govorilka
```

## Лицензия

MIT License — см. файл [LICENSE](LICENSE) (если добавлен).
