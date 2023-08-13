# import cv2 
# import numpy as np 
# from matplotlib import pyplot as plt

# face_cascade = cv2.CascadeClassifier( 
#     'haarcascade_frontalface_default.xml') # bring open source opencv

# image = cv2.imread('512 copy.jpg') # 이미지 불러오기
# # grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# faces = face_cascade.detectMultiScale(image, 1.03, 5) # 스케일

# print(faces.shape) #얼굴
# print("Number of faces detected: " + str(faces.shape[0])) #얼굴 개수


# for (x,y,w,h) in faces:
#     cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2) #박스 설정

# cv2.rectangle(image, ((0,image.shape[0] -25)), 
#         (270, image.shape[0]), (255,255,255), -1)#박스의 모양
# cv2.putText(image, "PinkWink test", (0,image.shape[0] -10), 
#         cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1) #그냥 pinkwink test 문구 넣어주는것

# plt.figure(figsize=(12,12)) #박스의 크기 (항상 정사각형)
# plt.imshow(image, cmap='gray')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

# [[(255,255,255),2,3,4], [1,2,3,4]]

import cv2
import numpy as np

# Yolo 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 이미지 가져오기
img = cv2.imread("phucket-soccer_orig copy.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
                   
# 정보를 화면에 표시
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# python flow --model ./cfg/my-tiny-yolo.cfg --labels ./labels.txt --trainer adam --dataset ../labelimages/ --annotation ../labelimages --train --summary ./logs --batch 3 --epoch 100 --save 50 --keep 5 --lr 1e-0 --load 1
