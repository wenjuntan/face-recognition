#-*-coding=utf-8-*-
import dlib
import numpy as np
import cv2
import json
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('2.dat')
facerec = dlib.face_recognition_model_v1('1.dat')
threshold = 0.54
def findNearestClassForImage(face_descriptor, faceLabel):
    temp =  face_descriptor - data
    e = np.linalg.norm(temp,axis=1,keepdims=True)
    min_distance = e.min() 
    print('distance: ', min_distance)
    if min_distance > threshold:
        return 'other'
    index = np.argmin(e)
    return faceLabel[index]
def recognition(img):
    dets = detector(img, 0)
    for k, d in enumerate(dets):
        
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        rec = dlib.rectangle(d.left(),d.top(),d.right(),d.bottom())
        print(rec.left(),rec.top(),rec.right(),rec.bottom())
        shape = sp(img, rec)
        face_descriptor = facerec.compute_face_descriptor(img, shape)        
        
        class_pre = findNearestClassForImage(face_descriptor, label)
        print(class_pre)
        cv2.rectangle(img, (rec.left(), rec.top()+10), (rec.right(), rec.bottom()), (0, 255, 0), 2)
        cv2.putText(img, class_pre , (rec.left(),rec.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow('image', img)
labelFile=open('label.txt','r')
label = json.load(labelFile)                                                   #载入本地人脸库的标签
labelFile.close()
    
data = np.loadtxt('faceData.txt',dtype=float)                                  #载入本地人脸特征向量

cap = cv2.VideoCapture('2.mp4')
fps = 10
size = (640,480)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4','v')
videoWriter = cv2.VideoWriter('3.avi', fourcc, fps, size)

while(1):
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    recognition(frame)
    videoWriter.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
videoWriter.release()
cv2.destroyAllWindows()
