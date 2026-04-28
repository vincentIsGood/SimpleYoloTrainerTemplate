import os
from ultralytics.trackers import BOTSORT, BYTETracker

import cv2
import pyautogui
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

##
# Tracking: Identify an object and assign an id (int) to it throughout the video
##

# Load the YOLO model
# model = YOLO("yolo26n.pt")

# Get latest trained model
latest_chkpt = os.listdir("runs/detect")
latest_chkpt.sort(key=lambda x: os.path.getmtime(f"runs/detect/{x}"))
model = YOLO(f"runs/detect/{latest_chkpt[-1]}/weights/best.pt")

def track_video():
    # Open the video file
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)
            handle_results(results)

            annotated_frame = results[0].plot() # gimme pixels array
            cv2.imshow("YOLO Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break # end of video

    # Release the video capture object and close the display window
    cap.release()

def track_screen():
    while True:
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        
        # Convert the screenshot to a numpy array
        frame = np.array(screenshot)
        
        # Convert colors from RGB to BGR (OpenCV format)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run YOLO tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        handle_results(results)

        annotated_frame = results[0].plot() # gimme pixels array
        cv2.imshow("YOLO Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def handle_results(results: list[Results]):
    if results[0].boxes is None:
        return

    if results[0].boxes.id is None:
        return
    
    ## Just in case you need to access tracker
    # tracker: BOTSORT | BYTETracker = model.predictor.trackers[0] # type: ignore
    # tracker.reset()
    ## OR
    # tracker.reset_id()

    for i in range(results[0].boxes.xyxy.shape[0]):
        classIndex = int(results[0].boxes.cls[i].item())
        classifiction = model.names[classIndex] # eg. "folder"
        box = results[0].boxes.xyxy[i]
        # ideally id == number of unique items tracked. 
        # But you need to track id yourself if you have different classes.
        id = results[0].boxes.id[i] # eg. 1
        x1, y1, x2, y2 = box.tolist()

        # do sth with the results...

track_screen()
cv2.destroyAllWindows()