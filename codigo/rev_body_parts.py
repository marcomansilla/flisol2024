import cv2

# Load the pre-trained HOG detector for pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Capture video from webcam (change 0 to the video file path for video file)
cap = cv2.VideoCapture(2)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame (optional, but can improve performance)
    frame = cv2.resize(frame, (640, 480))

    # Detect bodies in the frame
    bodies, _ = hog.detectMultiScale(frame)

    # Loop through the detected bodies
    for (x, y, w, h) in bodies:
        # Draw a rectangle around the detected body
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add labels for body parts (example: head)
        cv2.putText(frame, 'Head', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the result
    cv2.imshow('Detected Body Parts', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
