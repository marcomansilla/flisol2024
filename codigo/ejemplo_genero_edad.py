import cv2

# Load pre-trained models for face detection, gender, and age estimation
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")

# Function to detect faces, gender, and age in the video stream
def detect_faces_gender_age(video_capture):
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate through detected faces
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y + h, x:x + w]

            # Preprocess the face ROI for gender prediction
            gender_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Predict gender
            gender_net.setInput(gender_blob)
            gender_preds = gender_net.forward()
            gender = "Varon" if gender_preds[0][0] > gender_preds[0][1] else "Mujer"

            # Preprocess the face ROI for age prediction
            age_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Predict age
            age_net.setInput(age_blob)
            age_preds = age_net.forward()
            age = age_from_output(age_preds)

            # Draw rectangles around the detected faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = "{}, {}".format(gender, age)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection with Gender and Age', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    video_capture.release()
    cv2.destroyAllWindows()

# Function to extract age from the age prediction output
def age_from_output(age_preds):
    max_index = age_preds.argmax()
    predicted_age = max_index*10
    return predicted_age

# Main function
def main():
    # Initialize video capture from the default camera (0)
    video_capture = cv2.VideoCapture(2)

    # Check if the camera is opened successfully
    if not video_capture.isOpened():
        print("Error: Unable to open video capture.")
        return

    # Call the function to detect faces, gender, and age
    detect_faces_gender_age(video_capture)

if __name__ == "__main__":
    main()
