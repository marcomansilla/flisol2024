import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = DeepFace.build_model('Emotion')
emociones = {'angry': 'Enojo',
             'disgust': 'Disgusto',
             'fear': 'Miedo',
             'happy': 'Feliz',
             'sad': 'Triste',
             'surprise': 'Sorpresa',
             'neutral': 'Neutral'}

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        emotions = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
        predicted_emotion = emotions[0]['dominant_emotion']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emociones[predicted_emotion], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
