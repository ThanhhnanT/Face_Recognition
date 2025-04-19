import os
import face_recognition
import cv2
import pickle

root = 'images'
pathImageList = os.listdir(root)
studentId = []
imageList = []

for path in pathImageList:
    image = cv2.imread('{}/{}'.format(root, path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageList.append(image)
    studentId.append(path.split(".")[0])

print(len(image), len(studentId))

def ImageEncoding(imageList):
    encodeImages = []
    for image in imageList:
       encode = face_recognition.face_encodings(image)[0]
       encodeImages.append(encode)
    return encodeImages

imageEncode = ImageEncoding(imageList)
imageEncodeId = [imageEncode, studentId]
print(imageEncodeId)

file = open('imageEncodeId.p', 'wb')
pickle.dump(imageEncodeId, file)
file.close()
