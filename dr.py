#!/usr/bin/python
from __future__ import division  #change the / operator to mean true division throughout the module
import dlib
from imutils import face_utils
import cv2
import numpy as np
from scipy.spatial import distance as dist
import threading
import pygame
def start_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("z.ogg")
    pygame.mixer.music.play()


##Resize the captured image
def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA): # intterpolation for shrinking of image
    global ratio
    w, h = img.shape##returns tuple of no. of rows and col.
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
######shape object is converted to Numpy array
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)##it will create a multidimensional array with 68 rows and 2 cols with all 0s.
    for i in range(36,48):##coordinates of eyes
        coords[i] = (shape.part(i).x, shape.part(i).y)##fill the values from indices (36,48) of coordinate matrix with coordinates of image 
    return coords
####Fnction that accepts (x,y)- coordinates of given eye
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])# compute the euclidean distance between the vertical eye landmark
    B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
   
	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
    return ear
camera = cv2.VideoCapture(0)

predictor_path = 'shape_predictor_68_face_landmarks.dat_2'# from dlib

detector = dlib.get_frontal_face_detector()# detect frontal human faces in an image
predictor = dlib.shape_predictor(predictor_path)# detect the facial landmarks on the detected face
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]# mappings for the facial landmarks are encoded in FACIAL_LANDMARKS_IDXS dictionary in imutils library.
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
total=0
alarm=False
while True:
    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Color to gray image
    frame_resized = resize(frame_grey, width=120)# Call the function resize()

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
    dets = detector(frame_resized, 1)#No of faces detected
    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)# convert to numpy matrix
            leftEye= shape[lStart:lEnd]# coordinates of lfteye from numpy array(shape_to_np)
            rightEye= shape[rStart:rEnd]
            leftEAR= eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
           
	       
            
            
            if ear>.25:
                total=0
                alarm=False
                cv2.putText(frame, "Left Eye EAR ={}".format(leftEAR), (20, 30),cv2.FONT_HERSHEY_COMPLEX, 0.7, (100, 180, 200), 2)
                cv2.putText(frame, "Right Eye EAR ={}".format(rightEAR), (20, 70),cv2.FONT_HERSHEY_COMPLEX, 0.7, (100, 180, 200), 2)
                cv2.putText(frame, "Average EAR ={}".format(ear), (20, 110),cv2.FONT_HERSHEY_COMPLEX, 0.7, (100, 180, 200), 2)
            else:
                total+=1
                if total>4:
                    if not alarm:
                        alarm=True
                        d=threading.Thread(target=start_sound)
                        d.setDaemon(True)
                        d.start()
                        cv2.putText(frame, "Drowziness detected" ,(150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame, "Eyes close...drowsiness detected...Take a nap {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for (x, y) in shape:
                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 2, (50, 100, 200), -1)
    cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break
