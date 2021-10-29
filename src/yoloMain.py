import collections
from _datetime import datetime

import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase

location = "Sion East"
loc = {
    "Latitude": "19.044197669339596",
    "Longitude": "72.86488164388184"
}
services = ["Petrol", "Speed", "Air Filling", "Diesel", "CNG"]
cred = credentials.Certificate('./ServiceAccount.json')
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()
doc_ref = db.collection(u'Pumps').document(u'{}'.format(location))
doc_ref.set({
    u'Services': services,
    u'Location': loc
})
config = {
    "apiKey": "AIzaSyDef18Xg_Ri5Kvnc8VYabur4yVMdVvOAsw",
    "authDomain": "miniproject-442bd.firebaseapp.com",
    "projectId": "miniproject-442bd",
    "databaseURL": "https://miniproject-442bd-default-rtdb.asia-southeast1.firebasedatabase.app",
    "storageBucket": "miniproject-442bd.appspot.com",
    "messagingSenderId": "115419548616",
    "appId": "1:115419548616:web:18c046d397d5bfefcf8efd",
    "measurementId": "G-LCWK79KJ7C"
}
firebase = pyrebase.initialize_app(config)
database = firebase.database()
database.update({location: "Online"})
print('{} with {} at {}'.format(location, services, loc))
webcam = 0
cctv = 'rtsp://192.168.0.169/live/ch00_1'
n = 'night.mp4'
p = 'pump.mp4'
sd = 'sample_drive.mp4'
cap = cv2.VideoCapture(sd)

# 1 min -> maxlen 140

cs = collections.deque(maxlen=35)
bs = collections.deque(maxlen=35)
cs.append(0)
bs.append(0)
whT = 320
confThreshold = 0.6
nmsThreshold = 0.1
count = 0

classesFile = 'coco.names'
# classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
# print(len(classNames))

modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3-320.weights'


net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def findObjects(_outputs, _img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    cars = 0
    bikes = 0
    for i in indices:
        #i = i[0]
        box = bbox[i]
        if classNames[classIds[0]].upper() == 'CAR':
            cars += 1
        if (classNames[classIds[0]].upper() == 'MOTORBIKE') | (classNames[classIds[0]].upper() == 'PERSON'):
            bikes += 1
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {confs[i] * 100}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 3)
    # print("Cars:"+str(max(cs))+"\n"+"Bikes:"+str(max(bs)))
    cs.append(cars)
    bs.append(bikes)


while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Output layers name and index:
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs = net.forward(outputNames)
    # print(outputs[0].shape) #tiny:(192, 85)
    # print(outputs[1].shape) #tiny:(768, 85)
    findObjects(outputs, img)
    count += 1
    # print(count)
    if count % 35 == 0:
        # print(str(max(cs)))
        print('UPDATING DATA ' + str(int(count / 35)) + '\n CARS:\t' + str(max(cs)) + '\n BIKES:\t' + str(max(bs)))
        countUp = {"cars": max(cs), "bikes": max(bs)}
        upload = {
            location: countUp
        }
        database.update(upload)
        now = datetime.now()
        doc_upd = db.collection(u'Updates').document(u'{}'.format(now))
        doc_upd.set(
            {
                u'{}'.format(location): {
                    u'Bikes': max(bs),
                    u'Cars': max(cs)
                }
            }
        )

    cv2.imshow('YOLO', img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

database.update({location: "Offline"})
cap.release()
cv2.destroyAllWindows()
