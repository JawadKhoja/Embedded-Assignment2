# -*- coding: utf-8 -*-

import cv2
import torch
import time  # For measuring processing time


# Define the path for the video file
video_src = 'video.webm'  # Path to the video file

# Open the video file using OpenCV's VideoCapture class.
cap = cv2.VideoCapture(video_src)

# Load the pre-trained Haar cascade for car detection from OpenCV's default path
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Load the YOLOv5 model (the small YOLOv5 model 'yolov5s' is being used here)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the pre-trained YOLOv5 small model

# Check if the video was successfully opened
if not cap.isOpened():
    print(f"Error: Could not open video at {video_src}. Please check the file path.")
else:
    # Initialize counters for frames processed and objects detected.
    frame_count = 0
    total_objects_detected = 0

    # Start timing the processing.
    start_time = time.time()

    while True:
        # Read a frame from the video.
        ret, img = cap.read()
        
        # If no frame is read (end of video or error), break out of the loop.
        if ret is False:
            break
        
        frame_count += 1  # Increment the frame counter

        # Convert the image to grayscale for car detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform object detection (car detection) using Haar Cascade
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)

        # Draw rectangles around detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Perform object detection on the frame using YOLOv5
        results = yolo_model(img)  # Pass the frame to YOLOv5 for inference

        # Get the pandas dataframe containing detected objects
        detected_objects = results.pandas().xywh[0]  # The first dataframe in results

        # Count the number of objects detected by YOLOv5 in the frame
        num_objects_in_frame = len(detected_objects)
        total_objects_detected += num_objects_in_frame

        # Render the YOLOv5 results (bounding boxes, labels) on the frame
        results.render()  # This draws the results on the frame

        # Display the current frame with detected cars (Haar cascade) and objects (YOLO)
        cv2.imshow('Video', img)

        # Press 'Esc' (27) to exit the video display early
        if cv2.waitKey(33) == 27:
            break

    # Stop timing the processing.
    end_time = time.time()

    # Calculate and print the total processing time.
    total_time = end_time - start_time
    print(f"Total frames processed: {frame_count}")
    print(f"detected number of objects in frame: {detected_objects}")
   

    # Release the video capture and close the display window
    cap.release()
    cv2.destroyAllWindows()
