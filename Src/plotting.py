import matplotlib.pyplot as plt

def plot_detection_times(pupil_detection_count, no_pupil_detection_count, look_away_detected_count):
    # Create figure and axes
    fig, ax = plt.subplots()
    
    # Data for plotting
    categories = ['Pupil Detections', 'No Pupil Detections', 'Look Away Detections']
    counts = [pupil_detection_count, no_pupil_detection_count, look_away_detected_count]
    
    # Create bar chart
    ax.bar(categories, counts, color=['green', 'red', 'blue'])
    
    # Add labels and title
    ax.set_xlabel('Detection Categories')
    ax.set_ylabel('Count')
    ax.set_title('Pupil Detection Analysis')
    
    # Show plot
    plt.show()
