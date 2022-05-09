# Face Detection using Mediapipe
![face_detection_android_gpu](https://user-images.githubusercontent.com/61585411/167339934-0cf1336b-f72f-4028-a4cc-ddee13e5ec4e.gif)

## Steps Involved in Building Facial Landmarks Detection Application
- Importing all the essential libraries
- Reading a sample image
- Performing facial landmarks detection
- Drawing the results on the white frame image
<img src="https://user-images.githubusercontent.com/61585411/167333971-2c3f7a6a-bd2b-4cee-b924-55683218865b.gif">

## Step 1a: Import the libraries
```python
import cv2
import mediapipe as mp
import numpy as np
```
## Step 1b: Setting up a webcam (Windows)
```python
cap = cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)
```
It is quicker to get web cam live in Windows environment by adding cv2.CAP_DSHOW attribute.
## Step 1b: Setting up a webcam (Windows/Linux/Mac)
```python
cap = cv2.VideoCapture(0)
```
## Step 1c: Initialize the face_mesh class from the Mediapipe library
```python
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
```
## Step 2: Capturing the video feed using Webcam and create white frame to draw detected facial lankmarks
```python
while True:
    # Image
    ret, img = cap.read()
    if ret is not True:
        break
    height, width, _ = img.shape
    white_frame = np.zeros((height, width, 3), np.uint8)
    white_frame.fill(255)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
## Step 3: Performing facial landmarks detection and draw the results
```python
    # Facial landmarks
    result = face_mesh.process(rgb_img)
    try:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(white_frame, (x, y), 2, (100, 100, 0), -1)
    except Exception as e:
        pass
    cv2.imshow("Image", img)
    cv2.imshow("White", white_frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
        
cv2.destroyAllWindows()        
cap.release()
```
## References
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [Facial Landmarks Detection Using Mediapipe Library](https://www.analyticsvidhya.com/blog/2022/03/facial-landmarks-detection-using-mediapipe-library/)
