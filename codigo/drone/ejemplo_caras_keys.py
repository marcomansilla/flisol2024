from djitellopy import Tello
import cv2

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
    tello.takeoff()
    while True:
        frame = frame_reader.frame
        detect_faces(frame)
        cv2.imshow("Tello Video - Face Detection", frame)

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
