import pickle
import cv2
import os
import numpy as np
import face_recognition
from connectMongoDB import StudentModel
import requests

cap = cv2.VideoCapture(0)

cap.set(3, 449)
cap.set(4, 350)
curMatchIndex = -1
pathMode = os.listdir('interface')
pathMode = sorted(pathMode, key=lambda x: int(x.split('.')[0]))
print(pathMode)
imageMode = []
modeType = 0
for path in pathMode:
    image = cv2.imread(os.path.join('interface', path))
    image = cv2.resize(image, (361, 537))
    imageMode.append(image)

background = cv2.imread('background.jpeg')

# Loading imageEncode
file = open('imageEncodeId.p', 'rb')
imageEncode, studentId = pickle.load(file)
file.close()\
##### End loaD

while True:
    flag, frame = cap.read()
    if not flag:
        break
    frame_resized = cv2.resize(frame, (502, 306))
    imgS = cv2.resize(frame_resized, (0,0), None, 0.5, 0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    curFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, curFrame)
    if len(curFrame) > 0:
        # modeType = 1
        top, right, bottom, left = curFrame[0]
        cv2.rectangle(frame_resized, (right*2, top*2), (left*2,bottom*2), (255,0,0), 1)
    # else:
    #     modeType = 0
    for encodeFace in encodeCurFrame:
        match = face_recognition.compare_faces(imageEncode, encodeFace )
        distance = face_recognition.face_distance( imageEncode, encodeFace)
        print(match)
        print(distance)
        matchIndex = np.argmin(distance)
        cv2.putText(frame_resized, studentId[matchIndex], (right * 2, top * 2), fontFace=1, fontScale=1,
                    color=(0, 255, 0))
        print("matchIndex {} curIndex {}".format(matchIndex, curMatchIndex) )
        if match[matchIndex]:
            if matchIndex != curMatchIndex:
                curMatchIndex = matchIndex


                print(matchIndex, studentId[matchIndex])
                id = studentId[matchIndex]
                infor = StudentModel.find_one(
                    {'studentId': id},
                    {'fullName': 1, 'url': 1, '_id': 0, 'studentId': 1}
                )

                url = infor['url']
                fullName = infor['fullName']
                res = requests.get(url)
                avatar = np.asarray(bytearray(res.content), dtype=np.uint8)
                avatar = cv2.imdecode(avatar, cv2.IMREAD_COLOR)
                avatar = cv2.resize(avatar, (215,200))
                img =  imageMode[2].copy()
                cv2.putText(img, "StudentId: {}".format(id), (92,365), fontFace=1, fontScale=1,
                    color=(255, 255, 255) , thickness=2)
                cv2.putText(img, "FullName: {}".format(fullName), (92,435), fontFace=1, fontScale=1,
                    color=(255, 255, 255), thickness=2 )
                img[90:290, 75:290] = avatar
                modeType = 2
                print('Text success')
        else:
            modeType = 0
    if modeType != 2:
        background[0:537, 550: 911] = imageMode[modeType]
    else:
        background[0:537, 550: 911] = img
    background[164:470, 28:530] = frame_resized
    cv2.imshow('Webcam', background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
