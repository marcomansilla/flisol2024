from djitellopy import Tello
import cv2

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

def main():
    tello = Tello()
    tello.connect()
    tello.streamon()

    frame_reader = tello.get_frame_read()

    while True:
        frame = frame_reader.frame
        detect_faces(frame)  # Perform face detection
        cv2.imshow("Tello Video - Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
