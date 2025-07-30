import face_recognition
import cv2
import pyttsx3

engine = pyttsx3.init()

known_image = face_recognition.load_image_file("known_faces/known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for encoding in face_encodings:
        match = face_recognition.compare_faces([known_encoding], encoding)
        if match[0]:
            engine.say("Recognized face detected")
            engine.runAndWait()

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
