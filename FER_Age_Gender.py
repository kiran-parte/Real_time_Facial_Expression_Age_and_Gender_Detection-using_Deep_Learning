# python detect_faces_video.py --face face_detector --age age_detector --gender gender_detector --fer fer_model

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from keras.preprocessing import image
from keras.models import model_from_json

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
	help="path to age detector model directory")
ap.add_argument("-g", "--gender", required=True,
	help="path to gender detector model directory")
ap.add_argument("-w", "--fer", required=True,
	help="path to FER model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

prototxtPath = os.path.sep.join([args["face"], "face_deploy.prototxt"])
modelPath = os.path.sep.join([args["face"], "face_net.caffemodel"])
facenet = cv2.dnn.readNet(prototxtPath, modelPath)

genderProto = os.path.sep.join([args["gender"], "gender_deploy.prototxt"])
genderModel = os.path.sep.join([args["gender"], "gender_net.caffemodel"])
genderNet = cv2.dnn.readNet(genderModel, genderProto)

genderList = ['Male', 'Female']

ageProto = os.path.sep.join([args["age"], "age_deploy.prototxt"])
ageModel = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(ageModel, ageProto)

ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']

fer_model = os.path.sep.join([args["fer"], "fer_model.json"])
fer_weights = os.path.sep.join([args["fer"], "fer_model_weights.h5"])
model = model_from_json(open(fer_model,  "r",errors='ignore').read())
model.load_weights(fer_weights)

emotions = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

vs = VideoStream(1).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
 
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
 
	facenet.setInput(blob)
	detections = facenet.forward()

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence < args["confidence"]:
			continue

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		face = frame[startY:endY, startX:endX]
		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

		genderNet.setInput(faceBlob)
		genderPreds = genderNet.forward()

		ageNet.setInput(faceBlob)
		agePreds = ageNet.forward()

		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

		detected_face = cv2.resize(cv2.cvtColor(frame[int(startY):int(endY), int(startX):int(endX)], cv2.COLOR_BGR2GRAY), (48, 48))

		img_pixels = np.expand_dims(image.img_to_array(detected_face), axis = 0)
		img_pixels /= 255

		predictions = model.predict(img_pixels)

		label = "{}, {}, {}".format(genderList[genderPreds[0].argmax()], ageList[agePreds[0].argmax()], emotions[int(np.argmax(predictions))])

		cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()