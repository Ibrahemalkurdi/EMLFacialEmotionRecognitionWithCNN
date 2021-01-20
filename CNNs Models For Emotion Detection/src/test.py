import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np


config =  tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EMOTIONS_LIST = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
model_json_file = "JMyCourseraFacEmoReco.json"
model_weights_file = "model_mobilenetv2_weights_improved.h5"

with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_weights_file)
loaded_model._make_predict_function()

cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.stack((gray,)*3, axis=-1)
    #gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        # roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi_gray = cv2.resize(roi_gray, (96, 96), interpolation=cv2.INTER_AREA)
        # rect,face,image = face_detector(frame)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class

            preds = loaded_model.predict(roi)[0]
            label = EMOTIONS_LIST[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()