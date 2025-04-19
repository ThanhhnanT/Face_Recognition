import pickle
import cv2
import os
import numpy as np
import face_recognition
from connectMongoDB import StudentModel

cap = cv2.VideoCapture(0)

cap.set(3, 449)
cap.set(4, 350)

pathMode = os.listdir('interface')
imageMode = []
for path in pathMode:
    image = cv2.imread(os.path.join('interface', path))
    image = cv2.resize(image, (373, 504))
    imageMode.append(image)

background = cv2.imread('background.png')

# Loading imageEncode
file = open('imageEncodeId.p', 'rb')
imageEncode, studentId = pickle.load(file)
file.close()
print(len(imageEncode), studentId)

##### End loaD
while True:
    flag, frame = cap.read()
    if not flag:
        break
    frame_resized = cv2.resize(frame, (449, 350))
    imgS = cv2.resize(frame_resized, (0,0), None, 0.5, 0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    curFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, curFrame)
    if len(curFrame) > 0:
        top, right, bottom, left = curFrame[0]
        cv2.rectangle(frame_resized, (right*2, top*2), (left*2,bottom*2), (255,0,0), 1)

    for encodeFace in encodeCurFrame:
        match = face_recognition.compare_faces(imageEncode, encodeFace )
        distance = face_recognition.face_distance( imageEncode, encodeFace)
        matchIndex = np.argmin(distance)
        if match[matchIndex]:
            print(matchIndex, studentId[matchIndex])
            cv2.putText(frame_resized, studentId[matchIndex],  (right*2, top*2), fontFace = 1, fontScale= 1,color = (0,255,0))

    background[30:534, 530: 903] = imageMode[2]
    background[136:486, 45:45+449] = frame_resized
    cv2.imshow('Webcam', background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
