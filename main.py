import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage

path = 'images'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown = findEncodings(images)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        namelist = []
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

encodeListKnown = findEncodings(images)

def mail():
    EMAIL_ADDRESS = "chanduboddu199@gmail.com"
    EMAIL_PASSWORD = "7675894108"
    msg = EmailMessage()
    msg['Subject'] = 'Hello'
    msg['From'] = "chanduboddu199@gmail.com"
    msg['To'] = "chanduboddu758@gmail.com"
    msg.set_content('Today attendace')

    with open('Attendance.csv', 'rb') as f:
        file_data = f.read()
        file_name = f.name

    msg.add_attachment(file_data, maintype='application', subtype='octet_stream', filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

        smtp.send_message(msg)
        cv2.waitKey(0)


print('Encoding Complete')

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    success, img = cap.read()
    if (img is not None):
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex] > 0.5:
            name = classNames[matchIndex].upper()
        else:
            name = 'unknown'
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 255, 255), 2)
        markAttendance(name)
    if img is None:
        break
    mail()
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)