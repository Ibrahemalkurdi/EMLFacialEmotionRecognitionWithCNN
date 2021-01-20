# 1. read image
# 2. convert to gray scale
# 3. convert to uint8 range
# 4. threshold via otsu method
# 5. resize image
# 6. invert image to balck background
# 7. Feed into trained neural network 
# 8. print answer
# from skimage.io import imread
#from skimage.transform import resize
#from skimage import data, io
#from matplotlib import pyplot as plt
from skimage import img_as_ubyte		#convert float to uint8
from skimage.color import rgb2gray
import cv2
import datetime
import argparse
import imutils
import time
from time import sleep
from imutils.video import VideoStream
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np

face_classifier = cv2.CascadeClassifier('../resources/haarcascade_frontalface_default.xml')
# classifier = load_model('../resources/Emotion_little_vgg.h5')
classifier = load_model('../resources/emotion_face_mobilNet.h5')
#classifier = load_model('../resources/model_mobilenetv2_weights_improved.h5')
#classifier = load_model('../resources/emotion_face_mobilNet_NonTrainable.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

#model=load_model('mnist_trained_model.h5')		#import CNN model weight


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())


# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    # roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    # roi_gray = cv2.resize(roi_gray, (96, 96), interpolation=cv2.INTER_AREA)
    roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)
    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            

cv2.destroyAllWindows()
vs.stop()
