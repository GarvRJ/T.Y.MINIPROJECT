import collections
import pafy
import cv2
import numpy as np

webcam = 0
cctv = 'rtsp://192.168.0.169/live/ch00_1'
url = "https://www.youtube.com/watch?v=1EiC9bvVGnk"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
stream = best.url
n = 'night.mp4'
p = 'pump.mp4'
sd = 'sample_drive.mp4'
cap = cv2.VideoCapture(sd)

# 1 min -> maxlen 140
Services = ["petrol", "Speed", "Air Filling", "Diesel", "CNG"]
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
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def findObjects(outputs, img):
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
        i = i[0]
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
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
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

    cv2.imshow('YOLO', img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
