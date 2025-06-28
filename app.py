import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, AudioProcessorBase, WebRtcMode
import av
import cv2
import pickle
import numpy as np
import mediapipe as mp
import torch
import requests
import time
import threading
import websocket
import json
from urllib.parse import urlencode

# Page config
st.set_page_config(page_title="EchoSign", layout="wide")
st.title("EchoSign")

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# AssemblyAI streaming config
YOUR_API_KEY = "ce4ee1242ea544dfb103b7e98e824690"
CONNECTION_PARAMS = {
    "sample_rate": 16000,
    "format_turns": True,
}
API_ENDPOINT = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(CONNECTION_PARAMS)}"

# Load models
@st.cache_resource
def load_finger_model():
    return pickle.load(open("model.p", "rb"))["model"]
finger_model = load_finger_model()

@st.cache_resource
def load_dynamic_module():
    import asl_inference
    asl_inference.device = torch.device("cpu")
    return asl_inference
asl = load_dynamic_module()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)
labels_dict = {i: chr(ord('A') + i) for i in range(26)}

# Finger Processor
class FingerProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp = hands
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        H, W, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.mp.process(rgb)
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style(),
                                          mp_styles.get_default_hand_connections_style())
                xs = [lm.x for lm in hl.landmark]
                ys = [lm.y for lm in hl.landmark]
                data = []
                for x, y in zip(xs, ys):
                    data += [x - min(xs), y - min(ys)]
                if len(data) == 42:
                    p = finger_model.predict([np.array(data)])[0]
                    char = labels_dict[int(p)]
                    x1, y1 = int(min(xs)*W)-10, int(min(ys)*H)-10
                    x2, y2 = int(max(xs)*W)+10, int(max(ys)*H)+10
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0), 4)
                    cv2.putText(img, char, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Dynamic Processor
class DynamicProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = asl.mp_holistic.Holistic(static_image_mode=False,
                                                 min_detection_confidence=0.7,
                                                 min_tracking_confidence=0.7)
        self.buffer = []
        self.max_frames = 30
        self.last_text = ""
        self.display_count = 0
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
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
            cv2.putText(img, self.last_text,
                        (10, img.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0,255,0), 2)
            self.display_count -= 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Audio streaming Processor
global_stop = threading.Event()
class AudioStreamingProcessor(AudioProcessorBase):
    def __init__(self):
        # open WebSocket upfront
        self.ws = websocket.WebSocketApp(API_ENDPOINT,
                                         header={"Authorization": YOUR_API_KEY},
                                         on_message=self.on_message)
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def recv(self, frame: av.AudioFrame):
        pcm = frame.to_ndarray()           # raw PCM from browser
        self.ws.send(pcm.tobytes(), websocket.ABNF.OPCODE_BINARY)
        return None

    def on_message(self, ws, message):
        data = json.loads(message)
        if data.get("type")=="Turn" and data.get("turn_is_final"):
            txt = data["transcript"].strip()
            if txt:
                st.session_state.chat_history.append({"sender":"Hearing","message":txt})

# Layout
left, right = st.columns([2,1])
rtc_conf = {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}  
with left:
    st.header("Sign Recognition")
    mode = st.selectbox("Choose Mode:", ["Fingerspelling","Dynamic Sign"])
    if mode=="Fingerspelling":
        webrtc_streamer(key="fav", mode=WebRtcMode.SENDRECV,
                        video_processor_factory=FingerProcessor,
                        media_stream_constraints={"video":True},
                        rtc_configuration=rtc_conf)
    else:
        webrtc_streamer(key="dyn", mode=WebRtcMode.SENDRECV,
                        video_processor_factory=DynamicProcessor,
                        media_stream_constraints={"video":True},
                        rtc_configuration=rtc_conf)
with right:
    st.header("Voice to Chat")
    webrtc_streamer(key="aud", mode=WebRtcMode.SENDRECV,
                    audio_processor_factory=AudioStreamingProcessor,
                    media_stream_constraints={"audio":True},
                    rtc_configuration=rtc_conf)
    st.subheader("Chat History")
    cb=""
    for m in st.session_state.chat_history:
        c="#eef" if m['sender']=='Deaf' else '#ffe'
        cb+=f"<div style='background:{c};padding:8px;margin:4px;border-radius:4px;'><b>{m['sender']}:</b> {m['message']}</div>"
    st.markdown(f"<div style='height:400px;overflow-y:auto;border:1px solid #ccc;padding:4px;'>{cb}</div>",unsafe_allow_html=True)
