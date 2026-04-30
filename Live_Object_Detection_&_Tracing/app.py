import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import os
from datetime import datetime

# -------------------------------
# SETTINGS
# -------------------------------
ALERT_OBJECT = "person"   # Change this (e.g., "dog", "car")
SAVE_FOLDER = "detections"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# -------------------------------
# LOAD MODEL (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Debug panel
st.sidebar.header("🔧 Debug Panel")
debug_info = st.sidebar.empty()

# -------------------------------
# CALLBACK FUNCTION
# -------------------------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.track(img, persist=True, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    alert_triggered = False

    if results[0].boxes is not None:
        class_ids = results[0].boxes.cls.tolist()
        names = results[0].names
        counts = {}
        for cid in class_ids:
            label = names[int(cid)]
            counts[label] = counts.get(label, 0) + 1
            if label == ALERT_OBJECT:
                alert_triggered = True

        y_offset = 30
        for label, count in counts.items():
            cv2.putText(annotated_frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

    if alert_triggered:
        cv2.putText(annotated_frame, f"⚠ ALERT: {ALERT_OBJECT.upper()} DETECTED!",
                    (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        now = datetime.now()
        if not hasattr(video_frame_callback, "last_save"):
            video_frame_callback.last_save = now
        if (now - video_frame_callback.last_save).seconds >= 5:
            filename = f"{SAVE_FOLDER}/alert_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, annotated_frame)
            video_frame_callback.last_save = now

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# -------------------------------
# START STREAM WITH DEBUG INFO
# -------------------------------
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["turn:relay.metered.ca:80"], "username": "openai", "credential": "openai"},
        {"urls": ["turn:relay.metered.ca:443"], "username": "openai", "credential": "openai"},
        {"urls": ["turn:relay.metered.ca:443?transport=tcp"], "username": "openai", "credential": "openai"}
    ]
}

if "webrtc_started" not in st.session_state:
    st.session_state["webrtc_started"] = True
    ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )

    # Debug info
    if ctx and ctx.state.playing:
        debug_info.success("✅ WebRTC connection established, camera feed active.")
    else:
        debug_info.warning("⚠ Connection still pending... check camera permissions and TURN servers.")
