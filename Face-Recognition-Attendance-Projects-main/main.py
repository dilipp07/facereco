import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Define the path to the directory containing training images
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

# Load training images and class names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to find face encodings for a list of images
def findEncodings(images):
    encodeList = []
    for img in images:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            # Handle the case where no face is detected in the image
            print("No face detected in the image.")
        except Exception as e:
            print(f"Error: {e}")
    return encodeList

# Function to mark attendance in a CSV file
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Define the number of consecutive correct detections required for attendance update
consecutive_detections_required = 6
consecutive_correct_detections = [0] * len(classNames)
prev_detected_name = None

# Load the known face encodings
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for i, encodeFace in enumerate(encodesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            if name == prev_detected_name:
                consecutive_correct_detections[i] += 1
                if consecutive_correct_detections[i] >= consecutive_detections_required:
                    markAttendance(name)
                    consecutive_correct_detections[i] = 0
            else:
                prev_detected_name = name
                consecutive_correct_detections[i] = 1
        else:
            prev_detected_name = None
            consecutive_correct_detections[i] = 0

        y1, x2, y2, x1 = facesCurFrame[i]
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
