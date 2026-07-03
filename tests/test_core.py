import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest


class TestConfig:
    def test_default_values(self):
        from config import WHISPER_MODEL, WHISPER_LANGUAGE, VAD_THRESHOLD
        assert WHISPER_MODEL == "small"
        assert WHISPER_LANGUAGE == "ru"
        assert VAD_THRESHOLD == 0.5

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL", "tiny")
        import config
        monkeypatch.setattr(config, "WHISPER_MODEL", "tiny")
        assert config.WHISPER_MODEL == "tiny"


class TestLLMHistory:
    def test_build_messages(self):
        from llm import _build_messages, _history
        _history.clear()
        messages = _build_messages("Привет")
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Привет"

    def test_save_empty_answer(self):
        from llm import _save_answer, _history
        _history.clear()
        _save_answer("")
        assert len(_history) == 0

    def test_history_limit(self):
        from llm import _build_messages, _history, MAX_HISTORY
        _history.clear()
        for i in range(MAX_HISTORY + 5):
            _build_messages(f"Сообщение {i}")
        assert len(_history) <= MAX_HISTORY


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        from llm import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        assert not cb.is_open()

    def test_opens_after_failures(self):
        from llm import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open()

    def test_resets_after_success(self):
        from llm import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        for _ in range(2):
            cb.record_failure()
        cb.record_success()
        assert not cb.is_open()


class TestTTSSynthesize:
    def test_no_model_raises(self):
        from tts import synthesize
        import tts
        old_model = tts._model
        tts._model = None
        try:
            with pytest.raises(RuntimeError, match="не загружена"):
                synthesize("тест")
        finally:
            tts._model = old_model

    def test_empty_text_raises(self):
        from tts import synthesize
        import tts
        mock_model = MagicMock()
        old_model = tts._model
        tts._model = mock_model
        try:
            with pytest.raises(ValueError, match="Пустой текст"):
                synthesize("")
        finally:
            tts._model = old_model

    def test_none_text_raises(self):
        from tts import synthesize
        import tts
        mock_model = MagicMock()
        old_model = tts._model
        tts._model = mock_model
        try:
            with pytest.raises(ValueError, match="Пустой текст"):
                synthesize(None)
        finally:
            tts._model = old_model


class TestSTTTranscribe:
    def test_short_audio_returns_empty(self):
        try:
            from stt import transcribe
        except ImportError:
            pytest.skip("faster_whisper не установлен")
        import numpy as np
        from unittest.mock import patch
        with patch("stt._model", MagicMock()):
            result = transcribe(np.zeros(100, dtype=np.float32))
            assert result == ""

    def test_no_model_raises(self):
        try:
            from stt import transcribe
        except ImportError:
            pytest.skip("faster_whisper не установлен")
        import numpy as np
        import stt
        old_model = stt._model
        stt._model = None
        try:
            with pytest.raises(RuntimeError, match="Модель не загружена"):
                transcribe(np.zeros(1600, dtype=np.float32))
        finally:
            stt._model = old_model


class TestMetrics:
    def test_timing(self):
        from metrics import PipelineMetrics
        m = PipelineMetrics()
        with m.timing("test_stage"):
            pass
        stats = m.get_stats()
        assert "test_stage" in stats
        assert stats["test_stage"]["count"] == 1

    def test_reset(self):
        from metrics import PipelineMetrics
        m = PipelineMetrics()
        with m.timing("test"):
            pass
        m.reset()
        assert len(m.get_stats()) == 0


class TestEmotions:
    def test_extract_emotion_happy(self):
        from emotions import extract_emotion
        text, emotion = extract_emotion("Привет!\n[emotion:happy]")
        assert text == "Привет!"
        assert emotion == "happy"

    def test_extract_emotion_neutral_default(self):
        from emotions import extract_emotion
        text, emotion = extract_emotion("Привет!")
        assert text == "Привет!"
        assert emotion == "neutral"

    def test_extract_emotion_unknown(self):
        from emotions import extract_emotion
        text, emotion = extract_emotion("Текст [emotion:unknown]")
        assert emotion == "neutral"

    def test_get_emotion_params_happy(self):
        from emotions import get_emotion_params
        params = get_emotion_params("happy")
        assert "ParamMouthForm" in params
        assert params["ParamMouthForm"] > 0

    def test_get_emotion_params_neutral(self):
        from emotions import get_emotion_params
        params = get_emotion_params("neutral")
        assert params["ParamMouthForm"] == 0.0

    def test_lerp(self):
        from emotions import lerp
        assert lerp(0.0, 1.0, 0.0) == 0.0
        assert lerp(0.0, 1.0, 1.0) == 1.0
        assert lerp(0.0, 1.0, 0.5) == 0.5


class TestTTSCleanText:
    def test_clean_removes_emotion_tags(self):
        from tts import _clean_text
        result = _clean_text("Привет! [emotion:happy]")
        assert "[emotion" not in result
        assert "Привет" in result

    def test_clean_removes_markdown(self):
        from tts import _clean_text
        result = _clean_text("Это **важно** и _подчёркнуто_")
        assert "**" not in result
        assert "_" not in result
        assert "важно" in result

    def test_clean_collapses_spaces(self):
        from tts import _clean_text
        result = _clean_text("Слово    с    пробелами")
        assert "   " not in result

    def test_clean_empty_after(self):
        from tts import _clean_text
        result = _clean_text("[emotion:happy]")
        assert result == ""


class TestTTSNumbersAndEnglish:
    def test_numbers_to_words_simple(self):
        from tts import _numbers_to_words
        result = _numbers_to_words("42")
        assert "сорок" in result

    def test_numbers_to_words_in_sentence(self):
        from tts import _numbers_to_words
        result = _numbers_to_words("У меня 3 кошки")
        assert "три" in result
        assert "3" not in result

    def test_numbers_to_words_decimal(self):
        from tts import _numbers_to_words
        result = _numbers_to_words("3.14")
        assert "три" in result

    def test_numbers_to_words_no_change_text(self):
        from tts import _numbers_to_words
        result = _numbers_to_words("Привет мир")
        assert result == "Привет мир"

    def test_transliterate_simple(self):
        from tts import _transliterate_english
        result = _transliterate_english("Python")
        assert result == "Питон"

    def test_transliterate_multiple_words(self):
        from tts import _transliterate_english
        result = _transliterate_english("Docker container")
        assert "Докер" in result

    def test_transliterate_keeps_cyrillic(self):
        from tts import _transliterate_english
        result = _transliterate_english("Привет мир")
        assert result == "Привет мир"

    def test_transliterate_single_char_skipped(self):
        from tts import _transliterate_english
        result = _transliterate_english("A B")
        assert "A" in result

    def test_clean_text_numbers_converted(self):
        from tts import _clean_text
        result = _clean_text("Сегодня 25 декабря")
        assert "25" not in result
        assert "двадцать пять" in result

    def test_clean_text_english_converted(self):
        from tts import _clean_text
        result = _clean_text("Установи Docker")
        assert "Docker" not in result
        assert "Докер" in result


class TestTools:
    def test_execute_unknown_tool(self):
        from tools import execute_tool
        result = execute_tool("nonexistent", {})
        assert "Неизвестный" in result

    def test_get_time_returns_string(self):
        from tools import execute_tool
        result = execute_tool("get_time", {})
        assert isinstance(result, str)
        assert len(result) > 5

    def test_calculate_simple(self):
        from tools import execute_tool
        result = execute_tool("calculate", {"expression": "2 + 2"})
        assert "4" in result

    def test_calculate_division(self):
        from tools import execute_tool
        result = execute_tool("calculate", {"expression": "10 / 2"})
        assert "5" in result

    def test_calculate_blocks_imports(self):
        from tools import execute_tool
        result = execute_tool("calculate", {"expression": "__import__('os').system('ls')"})
        assert "Ошибка" in result

    def test_calculate_blocks_function_calls(self):
        from tools import execute_tool
        result = execute_tool("calculate", {"expression": "print(1)"})
        assert "запрещены" in result

    def test_set_timer_positive(self):
        from tools import execute_tool
        result = execute_tool("set_timer", {"minutes": 5, "label": "Тест"})
        assert "5" in result
        assert "установлен" in result

    def test_set_timer_negative(self):
        from tools import execute_tool
        result = execute_tool("set_timer", {"minutes": -1})
        assert "Ошибка" in result or "положительной" in result

    def test_system_info_returns_string(self):
        from tools import execute_tool
        result = execute_tool("system_info", {})
        assert isinstance(result, str)
        assert "ОС" in result or "OS" in result.upper()

    def test_tools_list_structure(self):
        from tools import TOOLS
        assert len(TOOLS) == 8
        for t in TOOLS:
            assert t["type"] == "function"
            assert "function" in t
            assert "name" in t["function"]
            assert "parameters" in t["function"]

    def test_open_app_unknown_returns_hint(self):
        from tools import _open_app
        result = _open_app("несуществующееприложение123")
        assert "не найдено" in result
        assert "Доступные" in result

    def test_open_app_known_notepad(self):
        from tools import _open_app
        result = _open_app("блокнот")
        assert "запущено" in result

    def test_open_app_calc(self):
        from tools import _open_app
        result = _open_app("калькулятор")
        assert "запущено" in result

    def test_open_app_case_insensitive(self):
        from tools import _open_app
        result = _open_app("БЛОКНОТ")
        assert "запущено" in result


class TestPythonSandbox:
    def test_run_python_simple(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "2 + 2"})
        assert "4" in result

    def test_run_python_print(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "print('hello world')"})
        assert "hello world" in result

    def test_run_python_multiline(self):
        from tools import execute_tool
        code = "x = 10\ny = 20\nprint(x + y)"
        result = execute_tool("run_python", {"code": code})
        assert "30" in result

    def test_run_python_syntax_error(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "def foo("})
        assert "Ошибка" in result

    def test_run_python_runtime_error(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "1 / 0"})
        assert "Ошибка" in result

    def test_run_python_empty_code(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": ""})
        assert "Ошибка" in result

    def test_run_python_no_output(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "x = 1"})
        assert "успешно" in result.lower() or "без вывода" in result.lower()

    def test_run_python_import_math(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "import math\nprint(math.pi)"})
        assert "3.14" in result

    def test_run_python_list_comprehension(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "print([x**2 for x in range(5)])"})
        assert "0" in result and "16" in result

    def test_run_python_via_execute_tool(self):
        import datetime
        from tools import execute_tool
        current_year = str(datetime.date.today().year)
        result = execute_tool("run_python", {"code": "import datetime\nprint(datetime.date.today().year)"})
        assert current_year in result

    def test_run_python_blocks_os_import(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "import os\nos.system('echo hacked')"})
        assert "запрещён" in result.lower() or "ошибка" in result.lower()

    def test_run_python_blocks_subprocess_import(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "import subprocess\nsubprocess.run(['echo','hacked'])"})
        assert "запрещён" in result.lower() or "ошибка" in result.lower()

    def test_run_python_blocks_shutil_import(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "import shutil\nshutil.rmtree('/')"})
        assert "запрещён" in result.lower() or "ошибка" in result.lower()

    def test_run_python_blocks_sys_import(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "import sys\nsys.exit()"})
        assert "запрещён" in result.lower() or "ошибка" in result.lower()

    def test_run_python_blocks_from_os_import(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "from os import system\nsystem('echo hacked')"})
        assert "запрещён" in result.lower() or "ошибка" in result.lower()

    def test_run_python_blocks_open_builtin(self):
        from tools import execute_tool
        result = execute_tool("run_python", {"code": "open('/etc/passwd')"})
        assert "ошибка" in result.lower() or "запрещён" in result.lower()


class TestProfiles:
    def test_init_creates_dirs(self):
        from profiles import init, PROFILES_DIR
        init()
        assert PROFILES_DIR.exists()

    def test_create_and_load_profile(self):
        from profiles import create_profile, load_profile, delete_profile
        create_profile("test_user", tts_voice="aidar")
        profile = load_profile("test_user")
        assert profile["name"] == "test_user"
        assert profile["tts_voice"] == "aidar"
        delete_profile("test_user")

    def test_switch_to_profile(self):
        from profiles import create_profile, switch_to, get_current, get_current_name, delete_profile
        create_profile("switch_test", tts_voice="baya")
        switch_to("switch_test")
        current = get_current()
        assert current is not None
        assert current["name"] == "switch_test"
        assert get_current_name() == "switch_test"
        delete_profile("switch_test")

    def test_list_profiles(self):
        from profiles import create_profile, list_profiles, delete_profile
        create_profile("list_test")
        profiles = list_profiles()
        assert "list_test" in profiles
        delete_profile("list_test")

    def test_delete_profile(self):
        from profiles import create_profile, delete_profile, get_profile_path
        create_profile("del_test")
        assert delete_profile("del_test") is True
        assert not get_profile_path("del_test").exists()

    def test_delete_nonexistent_profile(self):
        from profiles import delete_profile
        assert delete_profile("nonexistent_12345") is False

    def test_history_path(self):
        from profiles import create_profile, switch_to, get_history_path, delete_profile
        create_profile("hist_test")
        switch_to("hist_test")
        path = get_history_path()
        assert "hist_test" in path
        delete_profile("hist_test")

    def test_default_profile_values(self):
        from profiles import create_profile, load_profile, delete_profile
        create_profile("default_vals")
        profile = load_profile("default_vals")
        assert "system_prompt" in profile
        assert "greeting" in profile
        assert "tts_voice" in profile
        delete_profile("default_vals")


class TestFaceEngine:
    def test_load_lib_without_install(self):
        from face_engine import _load_lib
        result = _load_lib()
        assert isinstance(result, bool)

    def test_get_known_names_empty(self):
        from face_engine import get_known_names
        names = get_known_names()
        assert isinstance(names, list)

    def test_recognize_with_none_frame(self):
        from face_engine import recognize
        locations, names = recognize(None)
        assert locations == []
        assert names == []

    def test_capture_with_none_frame(self):
        from face_engine import capture_face_for_registration
        result = capture_face_for_registration(None)
        assert result is None

    def test_save_and_remove_face(self):
        import numpy as np
        from face_engine import save_face, remove_face, get_known_names, _load_lib
        if not _load_lib():
            return
        fake_encoding = np.zeros(128)
        save_face("test_face", fake_encoding)
        assert "test_face" in get_known_names()
        remove_face("test_face")
        assert "test_face" not in get_known_names()
