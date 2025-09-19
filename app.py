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

st.set_page_config(page_title="EchoSign")

st.title("EchoSign")

left_col, right_col = st.columns([3, 1])

# Load static fingerspelling model
@st.cache_resource
def load_finger_model():
    model = pickle.load(open("model.p", "rb"))["model"]
    return model

finger_model = load_finger_model()

# Load dynamic ASL model module (asl_inference.py)
@st.cache_resource
def load_dynamic_module():
    import asl_inference
    return asl_inference

asl = load_dynamic_module()

# label dict for fingerspelling
labels_dict = {i: chr(ord('A') + i) for i in range(26)}


# Processor for static fingerspelling
def create_finger_processor():
    class FingerProcessor(VideoProcessorBase):
        def __init__(self):
            # create a fresh Hands instance per processor, and local drawing/style refs
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
            # explicit cleanup to free native resources
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
                # return original frame if anything goes wrong
                return frame

    return FingerProcessor


# Processor for dynamic ASL sequence detection
def create_dynamic_processor():
    class DynamicProcessor(VideoProcessorBase):
        def __init__(self):
            # create a fresh Holistic instance per processor to isolate native resources
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
                # call processor's holistic so its internal state is used
                _ = self.holistic.process(rgb)
                landmarks = asl.extract_landmarks(img)
                if landmarks is not None:
                    self.buffer.append(landmarks)
                    img = asl.draw_landmarks(img, landmarks)
                # When enough frames collected, predict and set text
                if len(self.buffer) >= self.max_frames:
                    sign, conf = asl.predict_sign(self.buffer, asl.model, asl.device)
                    self.last_text = f"{sign} ({conf*100:.1f}%)"
                    self.display_count = self.max_frames
                    self.buffer.clear()
                # Draw last_text if within display window
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


# ---- UI + robust switching logic + callback-backed Start/Stop buttons below video ----

# put the mode selector in the left column
with left_col:
    mode = st.selectbox("Select mode:", ["Fingerspelling", "Dynamic Sign"])

# session state flags for play state and switching
if "playing_finger" not in st.session_state:
    st.session_state["playing_finger"] = False
if "playing_dynamic" not in st.session_state:
    st.session_state["playing_dynamic"] = False
if "current_mode" not in st.session_state:
    st.session_state["current_mode"] = None
if "switching" not in st.session_state:
    st.session_state["switching"] = False

# If user changed mode: request old streamer stop, mark switching, rerun to let teardown happen
if st.session_state["current_mode"] is not None and st.session_state["current_mode"] != mode and not st.session_state["switching"]:
    st.session_state["switching"] = True
    # stop previous playing sessions (do not auto-start the new one)
    if st.session_state["current_mode"] == "Fingerspelling":
        st.session_state["playing_finger"] = False
    else:
        st.session_state["playing_dynamic"] = False
    # set the new current mode; leave playing flags to user Start/Stop
    st.session_state["current_mode"] = mode
    st.experimental_rerun()

# On first visit, set current_mode but do NOT auto-start streams
if st.session_state["current_mode"] is None:
    st.session_state["current_mode"] = mode

# Reset switching flag after rerun
if st.session_state["switching"]:
    st.session_state["switching"] = False

# callback functions for immediate button actions
def start_fingerspelling():
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

# STUN server configuration
rtc_conf = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# show the correct streamer with desired_playing_state tied to session_state flags
with left_col:
    if mode == "Fingerspelling":
        webrtc_streamer(
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

        # Start / Stop buttons below the video using callbacks and fixed keys
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
            media_stream_constraints={
                "video": {"frameRate": {"ideal": 10, "max": 15}},
                "audio": False
            },
            async_processing=True,
            rtc_configuration=rtc_conf,
            desired_playing_state=st.session_state["playing_dynamic"],
        )

        # Start / Stop buttons below the video using callbacks and fixed keys
        cols = st.columns([1, 1])
        with cols[0]:
            st.button("Start Dynamic Sign", key="start_dynamic_btn", on_click=start_dynamic)
        with cols[1]:
            st.button("Stop Dynamic Sign", key="stop_dynamic_btn", on_click=stop_dynamic)

# Right column reserved for other UI (left intentionally minimal)
with right_col:
    st.write("")
