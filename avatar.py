import logging
import os
import tempfile
import time
import wave
import threading
import numpy as np
import pygame
import live2d.v3 as live2d
from live2d.v3 import StandardParams
from live2d.utils.lipsync import WavHandler
from config import (
    AVATAR_MODEL_PATH, AVATAR_WIDTH, AVATAR_HEIGHT,
    AVATAR_LIP_SYNC_MULT, AVATAR_BACKGROUND,
    FACE_RECOGNITION_ENABLED, CAMERA_INDEX,
)
from emotions import EmotionState, get_emotion_params, lerp

logger = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))


def glDrawbg(raw_rgb, w, h):
    from OpenGL.GL import (
        glRasterPos2f, glDrawPixels, GL_RGB, GL_UNSIGNED_BYTE,
        glMatrixMode, glLoadIdentity, glOrtho, GL_PROJECTION, GL_MODELVIEW,
    )
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, w, h, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRasterPos2f(0, 0)
    glDrawPixels(w, h, GL_RGB, GL_UNSIGNED_BYTE, raw_rgb)


_PARAM_MAP = {
    "ParamAngleX": StandardParams.ParamAngleX,
    "ParamAngleY": StandardParams.ParamAngleY,
    "ParamAngleZ": StandardParams.ParamAngleZ,
    "ParamEyeLOpen": StandardParams.ParamEyeLOpen,
    "ParamEyeROpen": StandardParams.ParamEyeROpen,
    "ParamEyeLSmile": StandardParams.ParamEyeLSmile,
    "ParamEyeRSmile": StandardParams.ParamEyeRSmile,
    "ParamEyeBallX": StandardParams.ParamEyeBallX,
    "ParamEyeBallY": StandardParams.ParamEyeBallY,
    "ParamBrowLY": StandardParams.ParamBrowLY,
    "ParamBrowRY": StandardParams.ParamBrowRY,
    "ParamBrowLAngle": StandardParams.ParamBrowLAngle,
    "ParamBrowRAngle": StandardParams.ParamBrowRAngle,
    "ParamBrowLForm": StandardParams.ParamBrowLForm,
    "ParamBrowRForm": StandardParams.ParamBrowRForm,
    "ParamMouthForm": StandardParams.ParamMouthForm,
    "ParamMouthOpenY": StandardParams.ParamMouthOpenY,
    "ParamCheek": StandardParams.ParamCheek,
}


class AvatarWindow:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._pending_wav = None
        self._pending_lock = threading.Lock()
        self._emotion_state = EmotionState(name="neutral", params=get_emotion_params("neutral"))
        self._emotion_lock = threading.Lock()
        self._current_param_values = {k: 0.0 for k in _PARAM_MAP}
        self._pending_emotion = None

        self._face_lock = threading.Lock()
        self._last_face_name = "Неизвестный"
        self._on_face_detected = None
        self._camera_frame = None
        self._capture_lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

    def set_emotion(self, emotion: str, duration: float = 0.5):
        with self._emotion_lock:
            self._pending_emotion = EmotionState(
                name=emotion,
                params=get_emotion_params(emotion),
                start_time=time.monotonic(),
                duration=duration,
            )
            logger.debug("Эмоция аватара: %s", emotion)

    def set_face_callback(self, callback):
        self._on_face_detected = callback

    def get_last_face(self) -> str:
        with self._face_lock:
            return self._last_face_name

    def get_camera_frame(self):
        with self._capture_lock:
            return self._camera_frame

    def play_audio_with_lipsync(self, audio: np.ndarray, sample_rate: int):
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        with wave.open(tmp_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        with self._pending_lock:
            self._pending_wav = tmp_path

    def _update_emotion(self, model):
        with self._emotion_lock:
            if self._pending_emotion is not None:
                now = time.monotonic()
                elapsed = now - self._pending_emotion.start_time
                t = min(1.0, elapsed / self._pending_emotion.duration) if self._pending_emotion.duration > 0 else 1.0

                for param_name, target_val in self._pending_emotion.params.items():
                    if param_name not in _PARAM_MAP:
                        continue
                    current = self._current_param_values.get(param_name, 0.0)
                    self._current_param_values[param_name] = lerp(current, target_val, t)

                if t >= 1.0:
                    self._emotion_state = self._pending_emotion
                    self._pending_emotion = None

        for param_name, value in self._current_param_values.items():
            if param_name in _PARAM_MAP:
                model.SetParameterValue(_PARAM_MAP[param_name], value, 1.0)

    def _init_camera(self):
        if not FACE_RECOGNITION_ENABLED:
            return None
        try:
            import cv2
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if cap.isOpened():
                logger.info("Камера %d открыта для аватара.", CAMERA_INDEX)
                return cap
            logger.warning("Камера %d не доступна.", CAMERA_INDEX)
        except ImportError:
            logger.warning("opencv-python не установлен. Камера отключена.")
        return None

    def _capture_camera(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    def _frame_to_surface(self, frame, display):
        try:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            surf = pygame.transform.scale(surf, display)
            return surf
        except Exception as e:
            logger.debug("Ошибка конвертации кадра: %s", e)
            return None

    def _run(self):
        pygame.init()
        live2d.init()

        display = (AVATAR_WIDTH, AVATAR_HEIGHT)
        pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Live2D Avatar")

        live2d.glInit()

        bg_surface = None
        if AVATAR_BACKGROUND and os.path.exists(AVATAR_BACKGROUND):
            bg_surface = pygame.image.load(AVATAR_BACKGROUND).convert()
            bg_surface = pygame.transform.scale(bg_surface, display)

        model = live2d.LAppModel()
        model.LoadModelJson(os.path.join(_DIR, AVATAR_MODEL_PATH))
        model.Resize(*display)

        wav_handler = WavHandler()
        current_wav = None
        is_playing = False

        cap = self._init_camera()
        face_engine = None
        if cap is not None:
            try:
                import face_engine as fe
                fe.load_known_faces()
                face_engine = fe
            except ImportError:
                logger.warning("face_engine не доступен.")

        frame_count = 0
        last_face_name = "Неизвестный"
        clock = pygame.time.Clock()

        while not self._stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop_event.set()

            with self._pending_lock:
                if self._pending_wav:
                    if current_wav and os.path.exists(current_wav):
                        os.remove(current_wav)
                    current_wav = self._pending_wav
                    self._pending_wav = None

            if current_wav and os.path.exists(current_wav):
                wav_handler.Start(current_wav)
                current_wav = None
                is_playing = True

            model.Update()
            self._update_emotion(model)

            if wav_handler.Update():
                rms = wav_handler.GetRms() * AVATAR_LIP_SYNC_MULT
                model.SetParameterValue(StandardParams.ParamMouthOpenY, rms)
                if rms < 0.01:
                    is_playing = False
            else:
                is_playing = False

            camera_frame = None
            if cap is not None:
                camera_frame = self._capture_camera(cap)
                if camera_frame is not None:
                    with self._capture_lock:
                        self._camera_frame = camera_frame

            if camera_frame is not None and face_engine is not None:
                frame_count += 1
                if frame_count % 10 == 0:
                    try:
                        _, names = face_engine.recognize(camera_frame)
                        if names:
                            face_name = names[0]
                            if face_name != "Неизвестный" and face_name != last_face_name:
                                last_face_name = face_name
                                with self._face_lock:
                                    self._last_face_name = face_name
                                if self._on_face_detected:
                                    try:
                                        self._on_face_detected(face_name)
                                    except Exception as e:
                                        logger.error("Ошибка callback лица: %s", e)
                            elif face_name == "Неизвестный":
                                with self._face_lock:
                                    self._last_face_name = "Неизвестный"
                    except Exception as e:
                        logger.debug("Ошибка распознавания: %s", e)

            if camera_frame is not None:
                cam_surface = self._frame_to_surface(camera_frame, display)
                if cam_surface:
                    raw = pygame.image.tostring(cam_surface, "RGB", True)
                    glDrawbg(raw, display[0], display[1])
                elif bg_surface:
                    raw = pygame.image.tostring(bg_surface, "RGB", True)
                    glDrawbg(raw, display[0], display[1])
                else:
                    live2d.clearBuffer()
            elif bg_surface:
                raw = pygame.image.tostring(bg_surface, "RGB", True)
                glDrawbg(raw, display[0], display[1])
            else:
                live2d.clearBuffer()

            model.Draw()
            pygame.display.flip()
            clock.tick(60 if is_playing else 30)

        if cap is not None:
            cap.release()
        if current_wav and os.path.exists(current_wav):
            os.remove(current_wav)

        live2d.dispose()
        pygame.quit()
