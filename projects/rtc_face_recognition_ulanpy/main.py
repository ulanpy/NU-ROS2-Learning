from src.__init__ import *
from src.face_recognizer import FaceRecognizer
import cv2 as cv

fr = FaceRecognizer(used_model=model, used_encoder=encoder, last_spoken_time=lst)
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from camera")
        break

    frame_count, faces = fr.process_new_frame(frame)
    if frame_count % PROCESS_INTERVAL == 0:
        modified_frame = fr.process_faces(frame, faces)
    else:
        modified_frame = fr.track_faces(frame, faces)

    if modified_frame is None:
        print("Error: Frame processing returned None")
        break

    # Directly display the processed frame
    cv.imshow('Video', modified_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

