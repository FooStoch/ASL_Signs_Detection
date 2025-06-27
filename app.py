import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import pickle
import numpy as np
import mediapipe as mp
import torch
import os

# Page config
st.set_page_config(page_title="EchoSign", layout="centered")
st.title("EchoSign")

# Load static fingerspelling model
@st.cache_resource
def load_finger_model():
    model = pickle.load(open("model.p", "rb"))["model"]
    return model
finger_model = load_finger_model()

# Load dynamic ASL model
@st.cache_resource
def load_dynamic_model():
    import asl_inference
    return asl_inference
asl = load_dynamic_model()

# Setup Mediapipe for static
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)
labels_dict = {i: chr(ord('A') + i) for i in range(26)}

# VideoProcessor for static fingerspelling
def create_finger_processor():
    class FingerProcessor(VideoProcessorBase):
        def __init__(self):
            self.processor = hands
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            H, W, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.processor.process(rgb)
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
                    char = ''
                    if len(data) == 42:
                        p = finger_model.predict([np.array(data)])[0]
                        char = labels_dict[int(p)]
                    x1, y1 = int(min(xs)*W)-10, int(min(ys)*H)-10
                    x2, y2 = int(max(xs)*W)+10, int(max(ys)*H)+10
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0), 4)
                    cv2.putText(img, char, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                1.3, (0,0,0), 3, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    return FingerProcessor

# VideoProcessor for dynamic ASL sequence detection
def create_dynamic_processor():
    class DynamicProcessor(VideoProcessorBase):
        def __init__(self):
            self.holistic = asl.mp_holistic.Holistic(
                static_image_mode=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7)
            self.buffer = []
            self.max_frames = 30
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)
            landmarks = asl.extract_landmarks_from_results(results, img.shape)
            if landmarks is not None:
                self.buffer.append(landmarks)
                # draw landmarks
                img = asl.draw_landmarks(img, landmarks)
            if len(self.buffer) >= self.max_frames:
                sign, conf = asl.predict_sequence(self.buffer, asl.model, asl.device)
                text = f"{sign} ({conf*100:.1f}%)"
                self.buffer.clear()
                cv2.putText(img, text, (10, img.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255,0,0), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    return DynamicProcessor

# UI
mode = st.selectbox("Select mode:", ["Fingerspelling", "Dynamic Sign"])

if mode == "Fingerspelling":
    webrtc_streamer(
        key="finger",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=create_finger_processor(),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True)
else:
    webrtc_streamer(
        key="dynamic",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=create_dynamic_processor(),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True)
