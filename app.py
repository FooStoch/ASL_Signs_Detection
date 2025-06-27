import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import pickle
import numpy as np
import mediapipe as mp
import torch
import os
import requests
import time

# Page config
st.set_page_config(page_title="EchoSign", layout="wide")
st.title("EchoSign")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# AssemblyAI settings
ASSEMBLYAI_API_KEY = "ce4ee1242ea544dfb103b7e98e824690"
BASE_URL = "https://api.assemblyai.com/v2"
HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

def transcribe_audio(file_bytes):
    upload_resp = requests.post(BASE_URL + "/upload", headers=HEADERS, data=file_bytes)
    audio_url = upload_resp.json()["upload_url"]
    data = {"audio_url": audio_url, "speech_model": "universal"}
    transcript_resp = requests.post(BASE_URL + "/transcript", json=data, headers=HEADERS)
    transcript_id = transcript_resp.json()["id"]
    polling = BASE_URL + f"/transcript/{transcript_id}"
    while True:
        result = requests.get(polling, headers=HEADERS).json()
        if result.get("status") == "completed":
            return result.get("text", "")
        if result.get("status") == "error":
            return "[Transcription Error]"
        time.sleep(1)

# Load fingerspelling model
@st.cache_resource
def load_finger_model():
    return pickle.load(open("model.p", "rb"))["model"]
finger_model = load_finger_model()

# Load dynamic ASL model
@st.cache_resource
def load_dynamic_model():
    import asl_inference
    # Ensure device is CPU
    asl_inference.device = torch.device("cpu")
    return asl_inference
asl = load_dynamic_model()

# Mediapipe for static
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)
labels_dict = {i: chr(ord("A") + i) for i in range(26)}

# FingerProcessor
def create_finger_processor():
    class FingerProcessor(VideoProcessorBase):
        def __init__(self):
            self.mp = hands
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")
                H, W, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = self.mp.process(rgb)
                if res.multi_hand_landmarks:
                    for hl in res.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img, hl, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )
                        xs = [lm.x for lm in hl.landmark]
                        ys = [lm.y for lm in hl.landmark]
                        data = []
                        for x, y in zip(xs, ys):
                            data += [x - min(xs), y - min(ys)]
                        char = ""
                        if len(data) == 42:
                            p = finger_model.predict([np.array(data)])[0]
                            char = labels_dict[int(p)]
                        x1, y1 = int(min(xs)*W)-10, int(min(ys)*H)-10
                        x2, y2 = int(max(xs)*W)+10, int(max(ys)*H)+10
                        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0), 4)
                        cv2.putText(img, char, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                    (0,0,0), 3, cv2.LINE_AA)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception as e:
                print("Error in FingerProcessor.recv():", e)
                return frame
    return FingerProcessor

# DynamicProcessor
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
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb)
                landmarks = asl.extract_landmarks(img)
                if landmarks is not None:
                    self.buffer.append(landmarks)
                    img = asl.draw_landmarks(img, landmarks)
                # once buffer full, predict
                if len(self.buffer) >= self.max_frames:
                    sign, conf = asl.predict_sign(self.buffer, asl.model, asl.device)
                    self.last_text = f"{sign} ({conf*100:.1f}%)"
                    self.display_count = self.max_frames
                    self.buffer.clear()
                if self.display_count > 0:
                    cv2.putText(img, self.last_text,
                                (10, img.shape[0]-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0,255,0), 2, cv2.LINE_AA)
                    self.display_count -= 1
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception as e:
                print("Error in DynamicProcessor.recv():", e)
                return frame
    return DynamicProcessor

# Layout
left_col, right_col = st.columns([2,1])
rtc_conf = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

with left_col:
    mode = st.selectbox("Select mode:", ["Fingerspelling", "Dynamic Sign"])
    if mode == "Fingerspelling":
        webrtc_streamer(
            key="finger", mode=WebRtcMode.SENDRECV,
            video_processor_factory=create_finger_processor(),
            media_stream_constraints={
                "video": {"frameRate": {"ideal":10,"max":15}},
                "audio": False
            },
            async_processing=True,
            rtc_configuration=rtc_conf
        )
    else:
        start = st.button("Start Dynamic Sign")
        stop  = st.button("Stop Dynamic Sign")
        if start: st.session_state.dynamic_on = True
        if stop:  st.session_state.dynamic_on = False
        if st.session_state.get("dynamic_on", False):
            webrtc_streamer(
                key="dynamic", mode=WebRtcMode.SENDRECV,
                video_processor_factory=create_dynamic_processor(),
                media_stream_constraints={
                    "video": {"frameRate": {"ideal":10,"max":15}},
                    "audio": False
                },
                async_processing=True,
                rtc_configuration=rtc_conf
            )

with right_col:
    st.subheader("Chat")
    chat_html = ""
    for msg in st.session_state.chat_history:
        color = "#eef" if msg["sender"]=="Deaf" else "#ffe"
        chat_html += (
            f"<div style='background:{color};"
            f"padding:8px;margin:4px;border-radius:4px;'>"
            f"<b>{msg['sender']}:</b> {msg['message']}</div>"
        )
    st.markdown(
        f"<div style='height:400px;overflow-y:auto;"
        f"border:1px solid #ccc;padding:4px;'>{chat_html}</div>",
        unsafe_allow_html=True
    )
    audio_file = st.file_uploader(
        "🎤 Upload or record audio", type=["wav","mp3"], key="audio"
    )
    if audio_file:
        with st.spinner("Transcribing…"):
            text = transcribe_audio(audio_file.read())
        st.session_state.chat_history.append({"sender":"Hearing","message":text})
        st.experimental_rerun()
