# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import pickle
import numpy as np
import mediapipe as mp
import torch
import os
import tempfile
from io import BytesIO
import streamlit.components.v1 as components
import gc
import time

# --- Page config ---
st.set_page_config(page_title="Computerpreter", layout="wide")
st.title("Computerpreter")

# optional CSS to let content expand more (keep if you used it before)
st.markdown(
    """
    <style>
    .appview-container .main .block-container{
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Layout ---
# left big, right narrow (you can change the ratio)
left_col, right_col = st.columns([1, 1])

# -------------------------
# Initialize session_state keys for recorder re-mounting & flags
# -------------------------
if "recorder_key" not in st.session_state:
    st.session_state["recorder_key"] = 0  # increment to remount recorder component
if "playing_finger" not in st.session_state:
    st.session_state["playing_finger"] = False
if "playing_dynamic" not in st.session_state:
    st.session_state["playing_dynamic"] = False
if "current_mode" not in st.session_state:
    st.session_state["current_mode"] = None
if "switching" not in st.session_state:
    st.session_state["switching"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "audio_data" not in st.session_state:
    st.session_state["audio_data"] = None

# -------------------------
# Models / cached resources
# -------------------------
@st.cache_resource
def load_finger_model():
    model = pickle.load(open("model.p", "rb"))["model"]
    return model

finger_model = load_finger_model()

# dynamic/sign module (keeps its own model inside; needed for video)
@st.cache_resource
def load_dynamic_module():
    import asl_inference
    return asl_inference

asl = load_dynamic_module()

# ---- Whisper loader: DO NOT cache as resource to avoid permanently holding memory.
# We load it on-demand inside the transcription handler and then delete it.
def load_whisper_model_local(size="base"):
    import whisper
    model = whisper.load_model(size)
    return model

# -------------------------
# Labels
# -------------------------
labels_dict = {i: chr(ord('A') + i) for i in range(26)}

# -------------------------
# Video processors (with cleanup)
# -------------------------
def create_finger_processor():
    class FingerProcessor(VideoProcessorBase):
        def __init__(self):
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )

        def __del__(self):
            try:
                if hasattr(self, "hands") and self.hands:
                    self.hands.close()
            except Exception:
                pass

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")
                H, W, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    for hl in res.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            img, hl, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_styles.get_default_hand_landmarks_style(),
                            self.mp_styles.get_default_hand_connections_style()
                        )
                        xs = [lm.x for lm in hl.landmark]
                        ys = [lm.y for lm in hl.landmark]
                        data = []
                        for x, y in zip(xs, ys):
                            data += [x - min(xs), y - min(ys)]
                        char = ''
                        if len(data) == 42:
                            p = finger_model.predict([np.array(data)])[0]
                            char = labels_dict[int(p)]
                        x1, y1 = int(min(xs)*W)-10, int(min(ys)*H)-10
                        x2, y2 = int(max(xs)*W)+10, int(max(ys)*H)+10
                        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0), 4)
                        cv2.putText(
                            img, char, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (0,0,0), 3, cv2.LINE_AA
                        )
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception:
                return frame
    return FingerProcessor

def create_dynamic_processor():
    class DynamicProcessor(VideoProcessorBase):
        def __init__(self):
            self.holistic = asl.mp_holistic.Holistic(
                static_image_mode=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.buffer = []
            self.max_frames = 30
            self.last_text = ""
            self.display_count = 0

        def __del__(self):
            try:
                if hasattr(self, "holistic") and self.holistic:
                    self.holistic.close()
            except Exception:
                pass

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _ = self.holistic.process(rgb)
                landmarks = asl.extract_landmarks(img)
                if landmarks is not None:
                    self.buffer.append(landmarks)
                    img = asl.draw_landmarks(img, landmarks)
                if len(self.buffer) >= self.max_frames:
                    sign, conf = asl.predict_sign(self.buffer, asl.model, asl.device)
                    self.last_text = f"{sign} ({conf*100:.1f}%)"
                    self.display_count = self.max_frames
                    self.buffer.clear()
                if self.display_count > 0:
                    cv2.putText(
                        img, self.last_text,
                        (10, img.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2, cv2.LINE_AA
                    )
                    self.display_count -= 1
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception:
                return frame
    return DynamicProcessor

# -------------------------
# Session control helpers
# -------------------------
def stop_all_streams_and_rerun():
    """Set playing flags to False, small pause and rerun so previous streams teardown."""
    st.session_state["playing_finger"] = False
    st.session_state["playing_dynamic"] = False
    # give threads/processes a moment to stop (helps webrtc teardown)
    time.sleep(0.2)
    st.experimental_rerun()

# Callbacks used for buttons (immediate effect)
def start_fingerspelling():
    # stop other stream and start this one
    st.session_state["playing_dynamic"] = False
    st.session_state["playing_finger"] = True
    st.session_state["current_mode"] = "Fingerspelling"

def stop_fingerspelling():
    st.session_state["playing_finger"] = False

def start_dynamic():
    st.session_state["playing_finger"] = False
    st.session_state["playing_dynamic"] = True
    st.session_state["current_mode"] = "Dynamic Sign"

def stop_dynamic():
    st.session_state["playing_dynamic"] = False

# -------------------------
# UI: left video + start/stop; right STT; bottom chat
# -------------------------

# --- Left column: video + start/stop ---
with left_col:
    mode = st.selectbox("Select mode:", ["Fingerspelling", "Dynamic Sign"])

# If user changed mode: stop previous streamer and let teardown happen
if st.session_state["current_mode"] is not None and st.session_state["current_mode"] != mode and not st.session_state["switching"]:
    st.session_state["switching"] = True
    # stop current playback so webrtc streamer can be torn down safely
    if st.session_state["current_mode"] == "Fingerspelling":
        st.session_state["playing_finger"] = False
    else:
        st.session_state["playing_dynamic"] = False
    st.session_state["current_mode"] = mode
    st.experimental_rerun()

if st.session_state["current_mode"] is None:
    st.session_state["current_mode"] = mode

if st.session_state["switching"]:
    st.session_state["switching"] = False

# STUN config
rtc_conf = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Lower capture resolution to reduce memory/CPU (helps reliability on constrained cloud instances)
video_constraints = {
    "video": {
        "width": {"ideal": 320},
        "height": {"ideal": 240},
        "frameRate": {"ideal": 10, "max": 15}
    },
    "audio": False
}

with left_col:
    if mode == "Fingerspelling":
        webrtc_streamer(
            key="finger",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=create_finger_processor(),
            media_stream_constraints=video_constraints,
            async_processing=True,
            rtc_configuration=rtc_conf,
            desired_playing_state=st.session_state["playing_finger"],
        )

        cols = st.columns([1, 1])
        with cols[0]:
            st.button("Start Fingerspelling", key="start_finger_btn", on_click=start_fingerspelling)
        with cols[1]:
            st.button("Stop Fingerspelling", key="stop_finger_btn", on_click=stop_fingerspelling)

    else:
        webrtc_streamer(
            key="dynamic",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=create_dynamic_processor(),
            media_stream_constraints=video_constraints,
            async_processing=True,
            rtc_configuration=rtc_conf,
            desired_playing_state=st.session_state["playing_dynamic"],
        )

        cols = st.columns([1, 1])
        with cols[0]:
            st.button("Start Dynamic Sign", key="start_dynamic_btn", on_click=start_dynamic)
        with cols[1]:
            st.button("Stop Dynamic Sign", key="stop_dynamic_btn", on_click=stop_dynamic)

# -------------------------
# Right column: speech-to-text tool (Whisper + audio recorder)
# -------------------------
with right_col:
    st.markdown("### Speech-to-Text")

    # Render audio recorder component with a key stored in session_state so we can remount it
    record_result = None
    try:
        # if pip package available, call it and give it a key so it remounts on key change
        from st_audiorec import st_audiorec  # type: ignore
        # when recorder_key changes, the component will be re-mounted and internal playback cleared
        record_result = st_audiorec(key=f"rec_{st.session_state['recorder_key']}")
    except Exception:
        # fallback to local component build (if you included the frontend)
        try:
            st_audiorec_comp = components.declare_component(
                "st_audiorec", path="st_audiorec/frontend/build"
            )
            record_result = st_audiorec_comp(key=f"rec_{st.session_state['recorder_key']}")
        except Exception as e:
            record_result = None
            st.warning("Audio recorder component not available. Install streamlit-audiorec or include the component frontend build. Error: {}".format(e))

    wav_bytes = None

    if isinstance(record_result, dict) and "arr" in record_result:
        with st.spinner("processing audio…"):
            ind, raw = zip(*record_result["arr"].items())
            ind = np.array(ind, dtype=int)
            raw = np.array(raw, dtype=int)
            sorted_bytes = raw[ind]
            stream = BytesIO(bytearray(int(v) & 0xFF for v in sorted_bytes))
            wav_bytes = stream.read()
    elif isinstance(record_result, (bytes, bytearray)):
        wav_bytes = bytes(record_result)

    # Save audio bytes into session_state for later transcription
    if wav_bytes is not None:
        st.session_state["audio_data"] = wav_bytes
        st.success("Recording captured")

    # Buttons to transcribe / clear audio
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        if st.button("Transcribe Audio"):
            if st.session_state.get("audio_data", None) is None:
                st.error("No recording found!")
            else:
                # write bytes to temp file and pass to whisper
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                try:
                    tmp.write(st.session_state["audio_data"])
                    tmp.flush()
                    tmp_path = tmp.name
                finally:
                    tmp.close()

                # --- Load whisper on demand (smaller models use far less memory) ---
                # If memory is problem for you, change "base" -> "small" -> "tiny"
                model_size = "base"
                try:
                    model = load_whisper_model_local(size=model_size)
                    with st.spinner("Transcribing..."):
                        transcription = model.transcribe(tmp_path)
                    text = transcription.get("text", "").strip()
                    if text:
                        st.session_state["chat_history"].append({"role": "user", "text": text})
                        st.success("Transcription added to chat")
                    else:
                        st.info("No text recognized")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                finally:
                    # cleanup model & temp file and free memory
                    try:
                        del model
                    except Exception:
                        pass
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
                    # free cached GPU memory if present
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # try to free python-level memory
                    gc.collect()

    with col_t2:
        # Clear Recording: clear stored bytes AND remount the recorder component which clears the component's internal UI
        if st.button("Clear Recording"):
            st.session_state["audio_data"] = None
            # bump key so component gets a fresh instance (removing any playback HTML it showed)
            st.session_state["recorder_key"] = st.session_state.get("recorder_key", 0) + 1
            st.info("Cleared audio buffer")
            # force rerun so the component remounts immediately
            st.experimental_rerun()

# --- Bottom chat area spanning the whole page ---
st.markdown("---")

# Header with Clear History button to the right
cols_header = st.columns([1, 8])
with cols_header[0]:
    st.markdown("## Chat & Transcripts (history)")
with cols_header[1]:
    if st.button("Clear History", key="clear_history_btn"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared")

# Input box (user can type messages)
user_input = st.chat_input("Send a message (or speak then transcribe):")
if user_input:
    st.session_state["chat_history"].append({"role": "user", "text": user_input})

# Render the chat history (preserves order)
for entry in st.session_state["chat_history"]:
    if entry["role"] == "user":
        st.chat_message("user").write(entry["text"])
    else:
        st.chat_message(entry.get("role", "assistant")).write(entry["text"])
