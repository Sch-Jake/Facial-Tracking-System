import matplotlib.pyplot as plt

def plot_detection_times(pupil_detection_count, no_pupil_detection_count, look_away_detected_count):
    """
    Plot a bar chart showing counts of different detection categories.

    Parameters:
    - pupil_detection_count (int): Total count of pupil detections.
    - no_pupil_detection_count (int): Total count of periods without pupil detection.
    - look_away_detected_count (int): Total count of instances where the pupil is detected near the eye outline,
                                      indicating the user may be looking away.

    Returns:
    - None
    """
   
   
    # Data for plotting
    categories = ['Pupil Detections', 'No Pupil Detections', 'Look Away Detections']
    counts = [pupil_detection_count, no_pupil_detection_count, look_away_detected_count]
    
    # Create bar chart
    plt.bar(categories, counts, color=['green', 'red', 'blue'])
    
    # Add labels and title
    plt.xlabel('Detection Categories')
    plt.ylabel('Count')
    plt.title('Pupil Detection Analysis')
    
    # Show plot
    plt.show()