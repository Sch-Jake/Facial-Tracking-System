def check_for_cheating(pupil_detection_count, look_away_detected_count, no_pupil_detection_count, total_duration, threshold=0.2):
    """
    Checks for cheating based on the ratio of look away detections to pupil detections,
    the no detection count, and the total duration for which the camera was open.
    
    Args:
        pupil_detection_count (int): The number of times pupils were detected.
        look_away_detected_count (int): The number of times look away was detected.
        no_pupil_detection_count (int): The number of times no pupils were detected.
        total_duration (float): The total duration for which the camera was open, in seconds.
        threshold (float): The ratio threshold for flagging cheating. Default is 0.2.
        
    Returns:
        bool: True if cheating is detected, False otherwise.
        str: Message with the result.
    """
    if pupil_detection_count == 0:
        return False, "No pupil detections to analyze."
    
    # Calculate the ratios
    look_away_ratio = look_away_detected_count / pupil_detection_count
    no_detection_ratio = no_pupil_detection_count / total_duration
    
    # Define thresholds for ratios
    look_away_threshold = threshold
    # Proper threshold to check for no detection
    no_detection_threshold = 1.5  

    if look_away_ratio > look_away_threshold or no_detection_ratio > no_detection_threshold:
        return True, f"Review tapes, Look away ratio is {look_away_ratio:.2f} and no detection ratio is {no_detection_ratio:.2f}, which exceeds the thresholds."
    else:
        return False, f"No cheating detected. Look away ratio is {look_away_ratio:.2f} and no detection ratio is {no_detection_ratio:.2f}, which are within the thresholds."
