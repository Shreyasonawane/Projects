import cv2
import pyttsx3
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0]

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        class_name = model.names[int(class_id)]
        engine.say(f"{class_name} detected")
        engine.runAndWait()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, class_name, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
