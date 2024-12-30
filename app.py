import streamlit as st
import cv2
import tempfile
from video_processor import VideoAnomalyDetector
import os
import matplotlib.pyplot as plt
import io
from PIL import Image

st.set_page_config(page_title="Video Anomaly Detection", layout="wide")

st.title("Video Anomaly Detection System")

# Initialize the anomaly detector
detector = VideoAnomalyDetector()

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Create a temporary file to store the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Create columns for video and graph
    col1, col2 = st.columns(2)

    # Create placeholders for video and graph
    with col1:
        st.subheader("Video Processing")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("Real-time Anomaly Graph")
        graph_placeholder = st.empty()
        
    # Process the video
    detector.process_video(video_path, video_placeholder, graph_placeholder)
        
    # Clean up the temporary file
    os.unlink(video_path)
