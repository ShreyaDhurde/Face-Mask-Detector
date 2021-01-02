import cv2
import numpy as np
from keras.models import load_model
my_model = load_model("my_model.h5")

results = {0:'No Mask',1:'Mask'}
colour = {0:(0,0,255),1:(0,255,0)}

rect_size = 4
cap = cv2.VideoCapture(0) 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (ret, frame) = cap.read()
    frame=cv2.flip(frame,1,1) 
    
    rerect_size = cv2.resize(frame, (frame.shape[1] // rect_size, frame.shape[0] // rect_size))

    faces = face_cascade.detectMultiScale(
        rerect_size,
        scaleFactor=1.1,
        minNeighbors=3,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = frame[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(150,150))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=my_model.predict(reshaped)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(frame,(x,y),(x+w,y+h),colour[label],2)
        cv2.putText(frame, results[label], (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour[label], 2)


    cv2.imshow('LIVE', frame)
    key = cv2.waitKey(1)
    
    if key == 27: 
        break

cap.release()

cv2.destroyAllWindows()