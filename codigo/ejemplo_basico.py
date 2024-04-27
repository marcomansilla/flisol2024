import cv2

cam = cv2.VideoCapture(2)

while True:
    stream, image = cam.read()
    cv2.imshow("Basic stream", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
