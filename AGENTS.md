# AGENTS.md — Bot_govorilka

## What this is

Russian voice assistant with Live2D avatar. All local (no cloud APIs). LM Studio on localhost:1234 for LLM.

## Quick commands

```bash
# Run the assistant
python main.py

# Run tests
python -m pytest tests/ -v

# Run a single test class
python -m pytest tests/test_core.py::TestTools -v

# Compile check all modules
python -c "import py_compile; [py_compile.compile(f, doraise=True) for f in ['config.py','tools.py','llm.py','main.py','avatar.py','face_engine.py','profiles.py','tts.py','stt.py','audio.py','emotions.py','metrics.py','wake_word.py']]"
```

## Critical dependencies

- `face-recognition` requires `dlib` which requires **Visual Studio C++ Build Tools** on Windows. If `pip install face-recognition` fails with CMake errors, install VS Build Tools first. Tests still pass without it (face_engine gracefully degrades).
- `torch` must be installed before `faster-whisper` and `silero-vad`.
- `silero-tts` package is NOT used — TTS loads via `torch.hub.load('snakers4/silero-models', 'silero_tts')` directly.

## Architecture

```
main.py          Entry point, orchestrates the full pipeline
├── audio.py     Recording (sounddevice) + VAD (Silero) + wake word
├── stt.py       faster-whisper STT with word-level streaming
├── llm.py       OpenAI client → LM Studio, tool calling, history
├── tools.py     8 tools: time, weather, timer, calc, search, sysinfo, open_app, (face registration)
├── tts.py       Silero TTS via torch.hub, 5 voices, number→words, English transliteration
├── avatar.py    Live2D + pygame/OpenGL, lip-sync, camera feed, face recognition
├── emotions.py  [emotion:xxx] tag extraction, Live2D parameter interpolation
├── face_engine.py  face_recognition wrapper, known faces database
├── profiles.py  User profiles with isolated history/voice/system_prompt
├── config.py    All settings via .env + python-dotenv, validate_config()
└── metrics.py   Pipeline timing context manager
```

## Key patterns

- **TTS text sanitization**: `_clean_text()` in `tts.py` must run before Silero TTS. It converts numbers to Russian words (`num2words`), transliterates English→Cyrillic, strips markdown/emotion tags. Silero crashes on special chars.
- **Emotion tags**: LLM outputs `[emotion:happy]` at end of response. `extract_emotion()` in `emotions.py` strips it. Main loop applies to avatar via `_avatar.set_emotion()`.
- **Tool calling flow**: `ask_with_tools()` makes non-streaming request with tools → if tool_calls, execute → one streaming request for text response. If empty after tools, falls back to `ask_stream()`.
- **Profile isolation**: Each face gets separate `history_{name}.json`, `system_prompt`, `tts_voice`. `profiles.py` manages switching.
- **Avatar thread**: Runs on daemon thread, communicates via thread-safe locks (`_pending_lock`, `_emotion_lock`, `_face_lock`). Never call avatar methods from main thread without thread safety.

## Config

All via `.env` file (see `.env.example`). Key settings:

| Setting | Default | Notes |
|---|---|---|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | Must be running |
| `WHISPER_MODEL` | `small` | tiny/base/small/medium/large-v3/turbo |
| `TTS_SPEAKER` | `kseniya` | kseniya/baya/xenia/aidar/eugene |
| `FACE_RECOGNITION_ENABLED` | `false` | Requires opencv-python + face-recognition |
| `WAKE_WORD_ENABLED` | `false` | Requires openwakeword |

## Tests

- 60 tests in `tests/test_core.py`
- Tests run without GPU, LM Studio, or camera — mocked/bypassed
- `face_recognition` tests skip gracefully if dlib not installed
- `faster_whisper` tests skip if not installed
- Run `python -m pytest tests/ -v` before committing

## Gotchas

- `model.speakers` in Silero TTS returns a **list** (not dict) — code handles both via `isinstance` check
- `ask_stream()` accepts optional `messages` param to avoid double-adding to history on fallback
- `_clean_tool_msg()` builds assistant message dict manually (not `model_dump()`) to avoid extra fields confusing LM Studio
- History is per-profile when face recognition is enabled, global `history.json` otherwise
