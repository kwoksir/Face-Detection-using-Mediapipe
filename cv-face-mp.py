import cv2
import mediapipe as mp
import time

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        results = face_detection.process(image)
        if results.detections:
            for id, detection in enumerate(results.detections):
                #mp_draw.draw_detection(image, detection)
                #print(id, detection)
                bBox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                cv2.rectangle(image, boundBox, (0, 255, 0), 3)
                cv2.putText(image, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow('Face Detection', image)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
