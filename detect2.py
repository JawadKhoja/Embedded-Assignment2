import os
import cv

# Path to Haar Cascade Classifier for vehicle detection
haar_cascade = 'haarcascade_cars.xml'  # Updated path to the downloaded Haar Cascade file
video_path = 'video.mp4'  # Input video file path

# Print the absolute path for debugging
print("Absolute path to haarcascade_cars.xml:", os.path.abspath(haar_cascade))

# Load the classifier and initialize the video capture object
car_cascade = cv2.CascadeClassifier(haar_cascade)

# Check if the classifier was loaded successfully
if car_cascade.empty():
    print("Error: Haar Cascade classifier could not be loaded. Check the XML file.")
    exit()

# Check if the video file exists and is accessible
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' does not exist.")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties (width, height, and fps)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the output video
output_video_path = 'output_video.webm'  # Output video file path
fourcc = cv2.VideoWriter_fourcc(*'vp80')  # Codec for WebM format (use 'XVID' for .avi)
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process only the first 200 frames
frame_count = 0

while True:
    # Read the current frame from the video
    ret, frames = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Could not read frame or end of video reached.")
        break

    # Ensure the frame is not empty
    if frames is None:
        print("Error: Frame is empty.")
        break

    # Convert the frame to grayscale (Haar Cascade works with grayscale images)
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect vehicles (cars) in the frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Output the detections for this frame
    print(f"Frame {frame_count+1}: Detected {len(cars)} car(s)")
    
    for (x, y, w, h) in cars:
        # Print the coordinates of each detected car
        print(f"  - Car detected at (x: {x}, y: {y}, w: {w}, h: {h})")
        
        # Draw rectangles around the detected cars
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Write the processed frame to the output video file
    out.write(frames)

    # Stop after processing the first 200 frames
    frame_count += 1
    if frame_count >= 200:
        break

# Release the video capture and writer objects
cap.release()
out.release()
