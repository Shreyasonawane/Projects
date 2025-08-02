import cv2
import os
from ultralytics import YOLO  # YOLOv8 object detection from Ultralytics
from tracking.deep_sort import DeepSort  # Deep SORT tracking module
from utils.helpers import draw_boxes     # Helper function to draw boxes on the frame

# Paths for input video, YOLO model weights, and output file
video_path = 'datasets/VisDrone/traffic_sample.mp4'
model_path = 'models/yolo/yolov8n.pt'
output_path = 'output.avi'

# Load the YOLOv8 model
model = YOLO(model_path)

# Initialize Deep SORT tracker
tracker = DeepSort()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video dimensions
width, height = int(cap.get(3)), int(cap.get(4))

# Define video writer to save output with bounding boxes
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

# Process video frame-by-frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop when video ends

    # Run object detection on the current frame
    results = model(frame)[0]

    detections = []
    # Extract bounding boxes from detection results
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box corners
        conf = float(box.conf)                 # Confidence score
        cls = int(box.cls)                     # Class ID (not used here)
        detections.append([x1, y1, x2 - x1, y2 - y1, conf])  # Convert to (x, y, w, h, conf)

    # Update tracker with current frame's detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Loop through all confirmed tracks and draw bounding boxes
    for track in tracks:
        if not track.is_confirmed():
            continue
        x, y, w, h = map(int, track.to_tlwh())  # Get bounding box
        track_id = track.track_id               # Get unique ID
        draw_boxes(frame, (x, y, w, h), track_id)  # Draw box and ID label

    # Save the annotated frame to the output video
    out.write(frame)

    # Show the frame in a window
    cv2.imshow("Tracking", frame)

    # Break loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources after processing
cap.release()
out.release()
cv2.destroyAllWindows()
