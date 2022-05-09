# Face Detection using Mediapipe
Google open-source MediaPipe was first introduced in June, 2019. It aims to make our life easy by providing some integrated computer vision and machine learning features. Media Pipe is a framework for building multimodal(e.g video,audio or any time series data),cross-platform (i.eAndroid,IOS,web,edge devices) applied ML pipelines. Mediapipe also facilitates the deployment of machine learning technology into demos and applications on a wide variety of different hardware platforms.

<img src="https://user-images.githubusercontent.com/61585411/167342300-c86e6c67-e05c-435e-ac08-ad9dd897decd.jpg" width=600>

## Steps Involved in Face Detection by using Mediapipe Framework
- Importing all the essential libraries
- Setting up webcam
- Initialize the face_detection class from the Mediapipe library
- Performing face detection
- Drawing the results with confidence level

## Step 1: Import the libraries
```python
import cv2
import mediapipe as mp
```
## Step 2: Setting up a webcam (Windows)
```python
cap = cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)
```
It is quicker to get web cam live in Windows environment by adding cv2.CAP_DSHOW attribute.
## Step 2: Setting up a webcam (Windows/Linux/Mac)
```python
cap = cv2.VideoCapture(0)
```
## Step 3: Initialize the face_detection class from the Mediapipe library
```python
mp_facedetector = mp.solutions.face_detection
#mp_draw = mp.solutions.drawing_utils
```
If you want to draw facial key points, uncomment the line. 
```python
#mp_draw = mp.solutions.drawing_utils
```
## Step 4: Performing face detection
```python
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
```
If you want to draw facial key points, uncomment the line. 
```python
#mp_draw.draw_detection(image, detection)
## References
- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html)
- [Face Detection Using Mediapipe In Python](https://mlhive.com/2021/12/face-detection-using-mediapipe-in-python)
