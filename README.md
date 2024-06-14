# Cheating Detection with Facial Recognition and Eye Tracking
## Overview
This application is designed to replicate real world cheating detection software that use facial regonition. This software analyzes video streams to identify potential cheating, such as looking away.

## Key Features 
### Facial Regonition
Utilizes Haar cascade classifiers to detect human faces within a video feed, enabling accurate tracking of facial features.
### Eye Tracking
Employs Haar cascade classifiers for detecing eyes and hough circle detection for localization of pupils. 
### Data visualization
Generates graphs illustrating pupil detection times, instances of looking away and overall engagement with their screen

## Gettin Started
Follow these instructions to set up and run the project on your machine
### Prerequisites
Make sure you have python 3.x installed on you machine

### Instillation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/Sch-Jake/Facial-Tracking-System.git
    cd cheating-recognition-software
    ```
2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```
### Running the Application
1. **Run the main application**:
    ```bash
    python main.py
    ```
2. **Exit the application and visualize the data**
    ```bash
    q
    ```
### Viewing Results
After running the application, results and plots will be generated, showcasing the time spent on and off the screen and potential cheating detections.

## Usage
1. **Detection**: The software will capture real-time video, detecting faces and eyes.
2. **Flagging Cheating**: If the software detects a lack of pupil detection for a significant duration or frequent look-aways, it will flag these events as potential cheating.

## Authors
Jake Schouten
## Acknowledgments
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)