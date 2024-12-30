# Video Anomaly Detection System

A real-time video anomaly detection system using YOLOv8 and Deep SORT for object tracking and anomaly detection.

## Features

- Real-time object detection using YOLOv8
- Object tracking with Deep SORT
- Dynamic anomaly score calculation
- Live video feed with detection visualization
- Real-time anomaly score graph
- User-friendly Streamlit interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/video-anomaly-detection.git
cd video-anomaly-detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Upload a video file through the interface

4. Watch the real-time processing with:
   - Object detection visualization
   - Live anomaly score graph

## Requirements

- Python 3.11
- OpenCV
- PyTorch
- YOLOv8
- Deep SORT
- Streamlit
- Other dependencies listed in requirements.txt

## Project Structure

- `app.py`: Main Streamlit application
- `video_processor.py`: Core video processing and anomaly detection logic
- `requirements.txt`: List of Python dependencies
- `README.md`: Project documentation

## License

MIT License
