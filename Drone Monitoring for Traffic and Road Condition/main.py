import cv2
import os
from ultralytics import YOLO
from tracking.deep_sort import DeepSort
from utils.helpers import draw_boxes

video_path = 'datasets/VisDrone/traffic_sample.mp4'
model_path = 'models/yolo/yolov8n.pt'
output_path = 'output.avi'

model = YOLO(model_path)
tracker = DeepSort()

cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cls = int(box.cls)
        detections.append([x1, y1, x2 - x1, y2 - y1, conf])

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        x, y, w, h = map(int, track.to_tlwh())
        track_id = track.track_id
        draw_boxes(frame, (x, y, w, h), track_id)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
