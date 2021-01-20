

from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

import numpy as np

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
import numpy as np
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

def ImagePreProcess(im_orig):
	im_gray = rgb2gray(im_orig)				#convert original to gray image
	#io.imshow(im_gray)
	#plt.show()
	img_gray_u8 = img_as_ubyte(im_gray)		# convert grey image to uint8
	#cv2.imshow("Window", img_gray_u8)
	#io.imshow(img_gray_u8)
	#plt.show()
	#Convert grayscale image to binary
	(thresh, im_bw) = cv2.threshold(img_gray_u8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	#cv2.imshow("Window", im_bw)
	#resize using opencv
	img_resized = cv2.resize(im_bw,(28,28))
	#cv2.imshow("Window", img_resized)
	############################################################
	#resize using sciikit
	#im_resize = resize(im,(28,28), mode='constant')
	#io.imshow(im_resize) 
	#plt.show()
	#cv2.imshow("Window", im_resize)
	##########################################################
	#invert image
	im_gray_invert = 255 - img_resized
	#cv2.imshow("Window", im_gray_invert)
	####################################
	im_final = im_gray_invert.reshape(1,28,28,1)
	# the below output is a array of possibility of respective digit
	ans = model.predict(im_final)
	print(ans)
	# choose the digit with greatest possibility as predicted dight
	ans = ans[0].tolist().index(max(ans[0].tolist()))
	print('DNN predicted digit is: ',ans)



def main():
	# loop over the frames from the video stream
	while True:
		try:
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
            #roi_gray = cv2.resize(roi_gray, (96, 96), interpolation=cv2.INTER_AREA)
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
                
cap.release()
cv2.destroyAllWindows()
"""
            # draw the timestamp on the frame
			timestamp = datetime.datetime.now()
			ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
			cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.35, (0, 0, 255), 1)
		 
			# show the frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
		 
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
				# do a bit of cleanup
				cv2.destroyAllWindows()
				vs.stop()
			elif key == ord("t"):
				cv2.imwrite("num.jpg", frame)  
				im_orig = cv2.imread("num.jpg")
				ImagePreProcess(im_orig)
			else:
				pass
				
		except KeyboardInterrupt:
			# do a bit of cleanup
			cv2.destroyAllWindows()
			vs.stop()
"""		

if __name__=="__main__":
	main()