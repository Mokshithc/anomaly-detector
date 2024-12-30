from video_processor import VideoAnomalyDetector
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Video Anomaly Detection')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='anomaly_graph.png', help='Path to output graph')
    args = parser.parse_args()

    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        return

    # Initialize detector
    detector = VideoAnomalyDetector()
    
    print("Processing video...")
    timestamps, scores = detector.process_video(args.video)
    
    print("Generating anomaly graph...")
    detector.plot_anomaly_scores(timestamps, args.output)
    
    # Get timestamps of anomalies
    anomaly_times = detector.get_anomaly_timestamps()
    if anomaly_times:
        print("\nAnomalies detected at the following timestamps (seconds):")
        for time in anomaly_times:
            print(f"- {time:.2f}")
    else:
        print("\nNo significant anomalies detected")
    
    print(f"\nAnalysis complete. Graph saved to {args.output}")

if __name__ == "__main__":
    main()
