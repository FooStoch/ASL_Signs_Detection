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

# --- Page config ---
st.set_page_config(page_title="Computerpreter", layout="wide")
st.title("Computerpreter")

# --- Layout: left = video, right = speech-to-text, bottom = chat ---
left_col, right_col = st.columns([1, 1])

# -------------------------
# Models / cached resources
# -------------------------
@st.cache_resource
def load_finger_model():
    model = pickle.load(open("model.p", "rb"))["model"]
    return model

finger_model = load_finger_model()

@st.cache_resource
def load_dynamic_module():
    import asl_inference
    return asl_inference

asl = load_dynamic_module()

# Whisper model loader (cached)
@st.cache_resource
def load_whisper_model():
    import whisper
    model = whisper.load_model("base")
    return model

# -------------------------
# Labels
# -------------------------
labels_dict = {i: chr(ord('A') + i) for i in range(26)}

# -------------------------
# Video processors (with best-effort session_state append)
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
                            # best-effort append to fingerspelling raw history
                            try:
                                if "fingerspelling_raw" in st.session_state:
                                    st.session_state["fingerspelling_raw"].append(char)
                            except Exception:
                                # processor thread may not have safe access; ignore
                                pass
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
            # collect predictions in this instance for best-effort retrieval
            self.predicted_signs = []

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
                    # store predicted sign in instance list
                    try:
                        self.predicted_signs.append(sign)
                    except Exception:
                        pass
                    # try to append to session_state dynamic_sequence (best-effort)
                    try:
                        if "dynamic_sequence" in st.session_state:
                            st.session_state["dynamic_sequence"].append(sign)
                    except Exception:
                        pass
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
# Session state defaults for video switching and chat / audio
# -------------------------
if "playing_finger" not in st.session_state:
    st.session_state["playing_finger"] = False
if "playing_dynamic" not in st.session_state:
    st.session_state["playing_dynamic"] = False
if "current_mode" not in st.session_state:
    st.session_state["current_mode"] = None
if "switching" not in st.session_state:
    st.session_state["switching"] = False

# Chat history & audio holders
if "chat_history" not in st.session_state:
    # each entry is a dict: {"role": "user"/"system"/"transcript", "text": "..."}
    st.session_state["chat_history"] = []
if "audio_data" not in st.session_state:
    st.session_state["audio_data"] = None

# Ensure dynamic_sequence and fingerspelling_raw exist
if "dynamic_sequence" not in st.session_state:
    st.session_state["dynamic_sequence"] = []
if "fingerspelling_raw" not in st.session_state:
    st.session_state["fingerspelling_raw"] = []

# -------------------------
# UI: left video + start/stop; right STT; bottom chat
# -------------------------

# --- Left column: video + start/stop ---
with left_col:
    mode = st.selectbox("Select mode:", ["Fingerspelling", "Dynamic Sign"])

# If user changed mode: stop previous streamer and let teardown happen
if st.session_state["current_mode"] is not None and st.session_state["current_mode"] != mode and not st.session_state["switching"]:
    st.session_state["switching"] = True
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

# Callbacks used for buttons (immediate effect)
def start_fingerspelling():
    # reset raw letter history
    st.session_state["fingerspelling_raw"] = []
    st.session_state["playing_dynamic"] = False
    st.session_state["playing_finger"] = True
    st.session_state["current_mode"] = "Fingerspelling"

def stop_fingerspelling():
    st.session_state["playing_finger"] = False
    # best-effort: combine processor-side data if available
    collected = []
    try:
        ctx = st.session_state.get("webrtc_ctx_finger")
        if ctx is not None and getattr(ctx, "video_processor", None) is not None:
            vp = ctx.video_processor
            collected += getattr(vp, "predicted_letters", []) or []
    except Exception:
        pass
    # include session_state raw list
    try:
        collected = (st.session_state.get("fingerspelling_raw", []) or []) + collected
    except Exception:
        collected = collected

    history = [c for c in collected if c]  # filter falsy
    # Apply sliding-window trimming logic as provided
    window_size = 10
    threshold = 6
    result = []
    prev_main = None

    # Only run if we have enough history; the algorithm will simply skip otherwise
    for i in range(len(history) - window_size + 1):
        window = history[i:i+window_size]
        counts = {}
        for letter in window:
            counts[letter] = counts.get(letter, 0) + 1
        main_letter = max(counts, key=counts.get)
        if counts[main_letter] >= threshold:
            if main_letter != prev_main:
                result.append(main_letter)
                prev_main = main_letter

    result_string = ''.join(result)
    # append chat message showing raw list and trimmed word
    st.session_state["chat_history"].append({
        "role": "assistant",
        "text": f"Fingerspelling detected (raw): {history}\nTrimmed result: {result_string}"
    })
    # clear raw history after processing
    st.session_state["fingerspelling_raw"] = []

def start_dynamic():
    # reset dynamic sequence storage
    st.session_state["dynamic_sequence"] = []
    st.session_state["playing_finger"] = False
    st.session_state["playing_dynamic"] = True
    st.session_state["current_mode"] = "Dynamic Sign"

def stop_dynamic():
    st.session_state["playing_dynamic"] = False
    # collect signs from webrtc processor context (best-effort)
    collected = []
    try:
        ctx = st.session_state.get("webrtc_ctx_dynamic")
        if ctx is not None and getattr(ctx, "video_processor", None) is not None:
            vp = ctx.video_processor
            collected += getattr(vp, "predicted_signs", []) or []
    except Exception:
        pass
    # Also include anything in st.session_state dynamic_sequence (safer)
    try:
        collected = (st.session_state.get("dynamic_sequence", []) or []) + collected
    except Exception:
        collected = collected
    # No dedup; just show full list
    final_seq = [s for s in collected if s]
    st.session_state["chat_history"].append({
        "role": "assistant",
        "text": f"Dynamic signs detected: {final_seq}"
    })
    # clear the dynamic_sequence to be ready for next start
    st.session_state["dynamic_sequence"] = []

# STUN config
rtc_conf = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

with left_col:
    if mode == "Fingerspelling":
        ctx_f = webrtc_streamer(
            key="finger",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=create_finger_processor(),
            media_stream_constraints={
                "video": {"frameRate": {"ideal": 10, "max": 15}},
                "audio": False
            },
            async_processing=True,
            rtc_configuration=rtc_conf,
            desired_playing_state=st.session_state["playing_finger"],
        )
        # store context for potential processor-side inspection
        st.session_state["webrtc_ctx_finger"] = ctx_f

        # Start / Stop buttons below the video using callbacks and fixed keys
        cols = st.columns([1, 1])
        with cols[0]:
            st.button("Start Fingerspelling", key="start_finger_btn", on_click=start_fingerspelling)
        with cols[1]:
            st.button("Stop Fingerspelling", key="stop_finger_btn", on_click=stop_fingerspelling)

    else:
        ctx_d = webrtc_streamer(
            key="dynamic",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=create_dynamic_processor(),
            media_stream_constraints={
                "video": {"frameRate": {"ideal": 10, "max": 15}},
                "audio": False
            },
            async_processing=True,
            rtc_configuration=rtc_conf,
            desired_playing_state=st.session_state["playing_dynamic"],
        )
        # store context for potential processor-side inspection
        st.session_state["webrtc_ctx_dynamic"] = ctx_d

        cols = st.columns([1, 1])
        with cols[0]:
            st.button("Start Dynamic Sign", key="start_dynamic_btn", on_click=start_dynamic)
        with cols[1]:
            st.button("Stop Dynamic Sign", key="stop_dynamic_btn", on_click=stop_dynamic)

# --- Right column: speech-to-text tool (Whisper + audio recorder) ---
with right_col:
    st.markdown("### Speech-to-Text")

    # Prefer the installed package API (if available). If not, fall back to the local component build.
    try:
        # recommended usage from the component package
        from st_audiorec import st_audiorec  # this should be provided by the pip/git package
        record_result = st_audiorec()
    except Exception:
        # fallback to local component path (only works if st_audiorec/frontend/build exists in your repo)
        try:
            st_audiorec_comp = components.declare_component(
                "st_audiorec", path="st_audiorec/frontend/build"
            )
            record_result = st_audiorec_comp()
        except Exception as e:
            record_result = None
            st.warning("Audio recorder component not available. Install streamlit-audio-recorder (in requirements) or include the component frontend build. Error: {}".format(e))

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
                tmp.write(st.session_state["audio_data"])
                tmp.flush()
                tmp_path = tmp.name
                tmp.close()

                model = load_whisper_model()
                with st.spinner("Transcribing..."):
                    transcription = model.transcribe(tmp_path)
                text = transcription.get("text", "").strip()
                if text:
                    st.session_state["chat_history"].append({"role": "user", "text": text})
                    st.success("Transcription added to chat")
                else:
                    st.info("No text recognized")
    with col_t2:
        st.write("Hit Reset when transcription is finished to save memory!")

# --- Bottom chat area spanning the whole page ---
st.markdown("---")

# Header with Clear History button to the right
cols_header = st.columns([1, 8])
with cols_header[0]:
    st.markdown("## Chat")
with cols_header[1]:
    # place the Clear History button on the right side of the header
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
        # we only use "assistant" role here for translations & replies
        st.chat_message(entry.get("role", "assistant")).write(entry["text"])
