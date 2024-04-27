from djitellopy import Tello
import cv2
from deepface import DeepFace

def detect_expressions(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_model = DeepFace.build_model('Emotion')
    emociones = {'angry': 'Enojo',
                 'disgust': 'Disgusto',
                 'fear': 'Miedo',
                 'happy': 'Feliz',
                 'sad': 'Triste',
                 'surprise': 'Sorpresa',
                 'neutral': 'Neutral'}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        emotions = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
        predicted_emotion = emotions[0]['dominant_emotion']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emociones[predicted_emotion], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def main():
    tello = Tello()
    tello.connect()
    tello.streamon()
    frame_reader = tello.get_frame_read()
    tello.takeoff()
    while True:
        frame = frame_reader.frame
        detect_expressions(frame)
        cv2.imshow("Tello Video - Expression Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            tello.land()
            break
        elif key == ord('w'):
            tello.move_forward(30)
        elif key == ord('s'):
            tello.move_back(30)
        elif key == ord('a'):
            tello.move_left(30)
        elif key == ord('d'):
            tello.move_right(30)
        elif key == ord('i'):
            try:
                tello.move_up(30)
            except Exception as e:
                print(f"Error moving up: {str(e)}")
        elif key == ord('k'):
            try:
                tello.move_down(30)
            except Exception as e:
                print(f"Error moving down: {str(e)}")
        elif key == ord('j'):
            tello.rotate_counter_clockwise(30)
        elif key == ord('l'):
            tello.rotate_clockwise(30)

    tello.streamoff()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
