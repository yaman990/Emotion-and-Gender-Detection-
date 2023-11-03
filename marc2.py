from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_classifier   = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


classifier_emotion =load_model(r'C:\Users\yaman\Desktop\MLproject\emotion_model.h5')
classifier_gender = load_model(r'C:\Users\yaman\Desktop\MLproject\gender_model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
gender_lable = {0:'female',1:'male'}



while True:
    _, frame = cap.read()
    label_emotion = []
    label_gender = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    


    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        facearea_gray = gray[y:y+h+10,x:x+w+10]
        facearea_gray = cv2.resize(facearea_gray,(48,48),interpolation=cv2.INTER_AREA)



        
        facearea = facearea_gray.astype('float')/255.0
        facearea = img_to_array(facearea)
        facearea = np.expand_dims(facearea,axis=0)

            #emotion output
        prediction_emotion = classifier_emotion.predict(facearea)[0]


        
        label_emotion=emotion_labels[prediction_emotion.argmax()]
        label_position_emotion = (x,y-10)
        cv2.putText(frame,label_emotion,label_position_emotion,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        # gender output
        prediction_gender = classifier_gender.predict(facearea)[0]
        
        label_gender=gender_lable[prediction_gender.argmax()]
        label_position_gender = (x,y-30)
        cv2.putText(frame,label_gender,label_position_gender,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        
    cv2.imshow('project',frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()