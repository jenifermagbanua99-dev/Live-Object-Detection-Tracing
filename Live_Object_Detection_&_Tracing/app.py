import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import os
from datetime import datetime

# -------------------------------
# SETTINGS
# -------------------------------
ALERT_OBJECT = "specific"   # Change this (e.g., "dog", "car")
SAVE_FOLDER = "detections"

os.makedirs(SAVE_FOLDER, exist_ok=True)

# -------------------------------
# LOAD MODEL (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Auto-downloads if not present

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# -------------------------------
# CALLBACK FUNCTION
# -------------------------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()

    alert_triggered = False

    # -------------------------------
    # OBJECT COUNTING + ALERT CHECK
    # -------------------------------
    if results[0].boxes is not None:
        class_ids = results[0].boxes.cls.tolist()
        names = results[0].names

        counts = {}

        for cid in class_ids:
            label = names[int(cid)]

            # Count objects
            counts[label] = counts.get(label, 0) + 1

            # Check alert condition
            if label == ALERT_OBJECT:
                alert_triggered = True

        # Display counts on screen
        y_offset = 30
        for label, count in counts.items():
            text = f"{label}: {count}"
            cv2.putText(
                annotated_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_offset += 30

    # -------------------------------
    # ALERT DISPLAY + CONTROLLED SAVE
    # -------------------------------
    if alert_triggered:
        cv2.putText(
            annotated_frame,
            f"⚠ ALERT: {ALERT_OBJECT.upper()} DETECTED!",
            (10, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        # Save only every 5 seconds (prevents storage overload)
        now = datetime.now()

        if not hasattr(video_frame_callback, "last_save"):
            video_frame_callback.last_save = now

        if (now - video_frame_callback.last_save).seconds >= 5:
            filename = f"{SAVE_FOLDER}/alert_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, annotated_frame)
            video_frame_callback.last_save = now

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# -------------------------------
# START STREAM
# -------------------------------
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
