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

            # placeholder buffer for future capture logic
            if "captured_finger" not in st.session_state:
                st.session_state["captured_finger"] = []

            # flag for whether we're currently recording letters (set by Start/Stop buttons)
            if "recording_finger" not in st.session_state:
                st.session_state["recording_finger"] = False

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

                            # If recording is enabled, append recognized char to captured list.
                            # (Simple: append every frame's prediction — you will later refine.)
                            try:
                                if st.session_state.get("recording_finger", False):
                                    st.session_state["captured_finger"].append(char)
                            except Exception:
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
                # return original frame if anything goes wrong
                return frame

    return FingerProcessor


# Processor for dynamic ASL sequence detection
def create_dynamic_processor():
    class DynamicProcessor(VideoProcessorBase):
        def __init__(self):
            # create a fresh Holistic instance per processor to isolate native resources
            # use asl.mp_holistic (module-level import inside asl_inference)
            self.holistic = asl.mp_holistic.Holistic(
                static_image_mode=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.buffer = []
            self.max_frames = 30
            self.last_text = ""
            self.display_count = 0

            # placeholder for captured sequences and recording flag
            if "captured_dynamic" not in st.session_state:
                st.session_state["captured_dynamic"] = []
            if "recording_dynamic" not in st.session_state:
                st.session_state["recording_dynamic"] = False

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
                    # If recording dynamic sequences, store the predicted sign
                    try:
                        if st.session_state.get("recording_dynamic", False):
                            st.session_state["captured_dynamic"].append((sign, conf))
                    except Exception:
                        pass
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


# ---- UI + robust switching logic with Start/Stop buttons ----

with left_col:
    mode = st.selectbox("Select mode:", ["Fingerspelling", "Dynamic Sign"])

# session state flags for play state, recording, and switching
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
    # stop previous playing sessions (but do not auto-start the new one)
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

# Provide Start / Stop buttons in the right column
with right_col:
    st.markdown("### Controls")
    if mode == "Fingerspelling":
        start = st.button("Start Fingerspelling")
        stop = st.button("Stop Fingerspelling")
        # Start: ensure dynamic is stopped and finger playing is True
        if start:
            st.session_state["playing_dynamic"] = False
            st.session_state["playing_finger"] = True
        if stop:
            st.session_state["playing_finger"] = False
        # Recording toggles (separate buttons) - optional quick toggles
        rec_start = st.button("Start Recording Fingerspelling")
        rec_stop = st.button("Stop Recording Fingerspelling")
        if rec_start:
            st.session_state["recording_finger"] = True
        if rec_stop:
            st.session_state["recording_finger"] = False

        # show captured placeholder
        st.write("Captured (fingerspelling) count:", len(st.session_state.get("captured_finger", [])))

    else:
        start = st.button("Start Dynamic Sign")
        stop = st.button("Stop Dynamic Sign")
        if start:
            st.session_state["playing_finger"] = False
            st.session_state["playing_dynamic"] = True
        if stop:
            st.session_state["playing_dynamic"] = False
        rec_start = st.button("Start Recording Dynamic")
        rec_stop = st.button("Stop Recording Dynamic")
        if rec_start:
            st.session_state["recording_dynamic"] = True
        if rec_stop:
            st.session_state["recording_dynamic"] = False

        st.write("Captured (dynamic) count:", len(st.session_state.get("captured_dynamic", [])))

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

# Small debug / info panel on right
with right_col:
    st.markdown("---")
    st.write("Mode:", st.session_state["current_mode"])
    st.write("Playing (finger):", st.session_state["playing_finger"])
    st.write("Playing (dynamic):", st.session_state["playing_dynamic"])
    st.write("Recording (finger):", st.session_state.get("recording_finger", False))
    st.write("Recording (dynamic):", st.session_state.get("recording_dynamic", False))
    # show a short preview of captured items
    st.write("Captured fingerspelling (last 20):", st.session_state.get("captured_finger", [])[-20:])
    st.write("Captured dynamic (last 10):", st.session_state.get("captured_dynamic", [])[-10:])
