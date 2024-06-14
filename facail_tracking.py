# import the opencv library 
import cv2 
  
  
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# define a video capture object 
video_capture = cv2.VideoCapture(0)

def detect_faces_and_eyes(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    
    eye_coords_text = "Eye Coordinates: "

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Increase minSize to avoid detecting nostrils
        eyes = eye_classifier.detectMultiScale(roi_gray, 1.1, 5, minSize=(125,125))
        for (ex, ey, ew, eh) in eyes:
            eye_center = (ex + ew // 2, ey + eh // 2)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            eye_coords_text += f"({eye_center[0]},{eye_center[1]})"
        cv2.putText(frame, eye_coords_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

while True:
    result, video_frame = video_capture.read()  # read frames from the video
   
     # terminate the loop if the frame is not read successfully
    if result is False:
        break
    # apply the function we created to the video frame
    processed_frame = detect_faces_and_eyes(video_frame)  
    # display the processed frame in a window named "Face and Eye Detection"
    cv2.imshow("Face and Eye Detection", processed_frame) 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()