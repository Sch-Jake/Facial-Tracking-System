from Src.facail_tracking import FaceEyeDetector
from Src.plotting import plot_detection_times
from Src.cheating_detection import check_for_cheating
from datetime import datetime

if __name__ == "__main__":
    #Initialize the face and eye detector
    detector = FaceEyeDetector()
    
    # Record the start time of the detection process
    start_time = datetime.now()
    #run the detector process
    detector.run()
    
    # Calculate the total duration
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # Plot the detection times and counts
    plot_detection_times(detector.pupil_detection_count, detector.no_pupil_detection_count, detector.look_away_detected_count)
    
    # Check for cheating
    cheating_detected, message = check_for_cheating(detector.pupil_detection_count, detector.look_away_detected_count, detector.no_pupil_detection_count, total_duration)
    #print response
    print(message)
