from djitellopy import Tello
import cv2

def main():
    tello = Tello()
    tello.connect()
    tello.streamon()

    while True:
        img = tello.get_frame_read().frame
        cv2.imshow("Tello Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
