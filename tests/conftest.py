import pytest


@pytest.fixture
def sample_audio():
    import numpy as np
    return np.random.randn(16000).astype(np.float32) * 0.1


@pytest.fixture
def mock_config(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("WHISPER_LANGUAGE", "ru")
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LM_STUDIO_MODEL", "")
