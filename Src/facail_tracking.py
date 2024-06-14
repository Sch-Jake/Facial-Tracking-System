import cv2
import numpy as np
from datetime import datetime

class FaceEyeDetector:
    def __init__(self, video_source=0):
        self.video_capture = cv2.VideoCapture(video_source)
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.last_pupil_detected_time = None
        self.pupil_detected_times = []
        self.pupil_not_detected_times = []
        self.pupil_detection_count = 0
        self.no_pupil_detection_count = 0
        self.look_away_detected_count = 0
    
    def detect_faces_and_eyes(self, frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        pupil_detected = False  # Flag to track pupil detection
        
        if len(faces) == 0:
            # No faces detected
            self.pupil_not_detected_times.append(0)
            self.no_pupil_detection_count += 1
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            roi_gray = gray_image[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Increase minSize to avoid detecting nostrils
            eyes = self.eye_classifier.detectMultiScale(roi_gray, 1.1, 5, minSize=(125,125))
            for (ex, ey, ew, eh) in eyes:
                eye_center = (ex + ew // 2, ey + eh // 2)
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                
                # Use HoughCircles for pupil detection within the eye ROI
                eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                circles = cv2.HoughCircles(eye_roi_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                           param1=50, param2=30, minRadius=5, maxRadius=30)
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (x_circle, y_circle, r) in circles:
                        # Draw a circle around the pupil
                        cv2.circle(roi_color, (ex + x_circle, ey + y_circle), r, (0, 255, 0), 4)
                        
                        # Draw a dot at the center of the circle (pupil center)
                        cv2.circle(roi_color, (ex + x_circle, ey + y_circle), 2, (0, 0, 255), 3)
                        
                        # Check if the pupil is close to the eye outline
                        if (x_circle - r < 0.1 * ew or x_circle + r > 0.9 * ew or
                            y_circle - r < 0.1 * eh or y_circle + r > 0.9 * eh):
                            self.look_away_detected_count += 1
                        
                        # Update detection times
                        current_time = datetime.now()
                        if self.last_pupil_detected_time is not None:
                            time_diff = current_time - self.last_pupil_detected_time
                            self.pupil_detected_times.append(time_diff.total_seconds())
                        self.last_pupil_detected_time = current_time
                        pupil_detected = True
        
        # If no pupil detected, update not detected times and count
        if not pupil_detected and self.last_pupil_detected_time is not None:
            time_diff = datetime.now() - self.last_pupil_detected_time
            self.pupil_not_detected_times.append(time_diff.total_seconds())
            self.last_pupil_detected_time = None
            self.no_pupil_detection_count += 1
        elif pupil_detected:
            self.pupil_detection_count += 1
        
        return frame
    
    def run(self):
        while True:
            # Capture frame-by-frame
            ret, video_frame = self.video_capture.read()
            if not ret:
                break
            
            # Apply the face and eye detection
            processed_frame = self.detect_faces_and_eyes(video_frame)
            
            # Display the processed frame
            cv2.imshow('Face and Eye Detection', processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the video capture object and close all windows
        self.video_capture.release()
        cv2.destroyAllWindows()
