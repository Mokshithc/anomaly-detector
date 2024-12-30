import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import streamlit as st
import io
from PIL import Image

class VideoAnomalyDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.tracker = DeepSort(max_age=5)
        self.frame_history = []
        self.anomaly_scores = []

    def process_video(self, video_path, video_placeholder, graph_placeholder):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Run YOLOv8 tracking
            results = self.model.track(frame, persist=True)
            
            # Get boxes and track IDs
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # Calculate anomaly score based on object movements
                anomaly_score = self._calculate_anomaly_score(boxes, track_ids)
                self.anomaly_scores.append(anomaly_score)
                
                # Draw boxes and IDs on frame
                annotated_frame = results[0].plot()
                
                # Add anomaly score to frame
                cv2.putText(annotated_frame, f"Anomaly Score: {anomaly_score:.2f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                annotated_frame = frame
                self.anomaly_scores.append(0)

            # Display the frame in Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

            # Update the graph in real-time
            self._update_graph(graph_placeholder, frame_count)

        cap.release()
        
    def _calculate_anomaly_score(self, boxes, track_ids):
        if len(boxes) == 0:
            return 0
            
        # Calculate average movement speed
        current_positions = {id: box for id, box in zip(track_ids, boxes)}
        
        if len(self.frame_history) > 0:
            prev_positions = self.frame_history[-1]
            speeds = []
            
            for track_id in current_positions:
                if track_id in prev_positions:
                    curr_box = current_positions[track_id]
                    prev_box = prev_positions[track_id]
                    speed = np.sqrt((curr_box[0] - prev_box[0])**2 + (curr_box[1] - prev_box[1])**2)
                    speeds.append(speed)
            
            avg_speed = np.mean(speeds) if speeds else 0
            anomaly_score = avg_speed * len(boxes)
        else:
            anomaly_score = len(boxes)
            
        self.frame_history.append(current_positions)
        return anomaly_score
        
    def _update_graph(self, graph_placeholder, frame_count):
        # Create the graph
        plt.figure(figsize=(10, 6))
        plt.plot(self.anomaly_scores)
        plt.title('Real-time Anomaly Scores')
        plt.xlabel('Frame Number')
        plt.ylabel('Anomaly Score')
        plt.grid(True)
        
        # Set x-axis limit to show recent history
        if frame_count > 100:
            plt.xlim(frame_count - 100, frame_count)
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_image = Image.open(buf)
        
        # Display in Streamlit
        graph_placeholder.image(graph_image, use_column_width=True)
        
        # Close the figure to free memory
        plt.close()

def main():
    st.title("Video Anomaly Detector")
    video_path = st.text_input("Enter video path")
    if st.button("Start Analysis"):
        detector = VideoAnomalyDetector()
        video_placeholder = st.empty()
        graph_placeholder = st.empty()
        detector.process_video(video_path, video_placeholder, graph_placeholder)

if __name__ == "__main__":
    main()
