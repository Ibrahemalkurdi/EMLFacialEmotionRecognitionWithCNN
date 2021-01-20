import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

from picamera.array import PiRGBArray
from picamera import PiCamera

import time
import cv2
import numpy as np


model_json_file = "../resources/JMyCourseraFacEmoReco.json"
model_weights_file = "../resources/model_mobilenetv2_weights_improved.h5"

config =  tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)


with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
model.load_weights(model_weights_file)
model._make_predict_function()

face_classifier = cv2.CascadeClassifier('../resources/haarcascade_frontalface_default.xml')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	#print(img)
	faces = face_classifier.detectMultiScale(gray, 1.3, 5)
	if len(faces) != 0:
            for(x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_img = gray[y:y + h, x:x + w]
                roi_img = cv2.resize(roi_img, (96, 96), interpolation=cv2.INTER_AREA)
                print(roi_img.shape)
                if np.sum([roi_img]) != 0:
                    roi = roi_img.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    print(roi.shape)
                    
                    preds = model.predict(roi)[0]
                    label = class_labels[preds.argmax()]
                    label_position = (x, y)
                    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    
            #print(faces)
	# show the frame
	cv2.imshow("Emotion Detector", image)
	key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
            break
cv2.destroyAllWindows()