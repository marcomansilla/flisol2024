import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break
            cv2.imshow('Live Video Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
