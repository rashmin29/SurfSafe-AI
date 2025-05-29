from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
import os
import torch
import uuid
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from flask_socketio import SocketIO
import sqlite3
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the YOLOv8 model (only once)
model = YOLO('yolov8_custom_shark_detection_V1.pt')
cancel_processing = False  # Global flag for cancellation
executor = ThreadPoolExecutor(max_workers=4)  # Thread pool for multithreading

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('sharkVideoStore.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS sharkVideoStore
                    (email TEXT, input_video TEXT, output_video TEXT, original_filename TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

# Route for the home page (video upload form)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video processing and streaming
@app.route('/process', methods=['POST'])
def process_video():
    global cancel_processing
    cancel_processing = False  # Reset the flag at the start of processing
    
    # Get email from the form
    email = request.form.get('email')
    if 'video' not in request.files or not email:
        return redirect(url_for('index'))

    # Get the original filename
    video_file = request.files['video']
    original_filename = video_file.filename

    # Create unique filename based on email and timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{email}_{timestamp}_{uuid.uuid4().hex}.mp4"

    # Save input video to static/in
    video_path = os.path.join('static/in', unique_filename)
    video_file.save(video_path)

    # Save the information in the database
    conn = sqlite3.connect('sharkVideoStore.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sharkVideoStore (email, input_video, original_filename, timestamp) VALUES (?, ?, ?, ?)", 
                   (email, unique_filename, original_filename, timestamp))
    conn.commit()
    conn.close()

    return render_template('processing.html', email=email, video_name=unique_filename)

# Route to stream the video feed
@app.route('/video_feed/<video_name>')
def video_feed(video_name):
    return Response(generate_frames(video_name), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle cancellation
@app.route('/cancel_processing', methods=['POST'])
def cancel_processing_route():
    global cancel_processing
    cancel_processing = True
    
    return '', 200

def process_frame(frame):
    # Perform YOLOv8 detection on the frame
    results = model(frame)

    # Get the detections from the results
    detections = results[0].boxes

    # If no detections or the label is not 'shark', return the original frame without annotations
    annotated_frame = frame.copy()
    
    for detection in detections:
        label = detection.cls  # Class label for the detected object
        label_name = model.names[int(label)]  # Get the human-readable label name from model
        
        if label_name == "shark":
            # Annotate the frame if the label is 'shark'
            annotated_frame = results[0].plot()  # Add annotations
            break  # Exit after the first shark detection (optional)
    
    # Encode the frame in JPEG format
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    return buffer.tobytes()

def generate_frames(video_name):
    global cancel_processing
    video_path = os.path.join('static/in', video_name)
    output_video_path = os.path.join('static/out', video_name)

    # Create a thumbnail for the output video
    create_thumbnail(video_path, video_name)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

    try:
        while cap.isOpened():
            if cancel_processing:
                print("Processing canceled by user.")
                break  # Stop processing if cancel button was clicked

            ret, frame = cap.read()
            if not ret:
                break

            # Submit the frame for YOLO processing in a separate thread
            future = executor.submit(process_frame, frame)

            # Get the processed frame from the future object
            frame_bytes = future.result()

            # Convert the byte array to a NumPy array for decoding
            frame_array = np.frombuffer(frame_bytes, np.uint8)

            # Decode the image (convert bytes to an OpenCV-readable format)
            annotated_frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if annotated_frame is None:
                print("Error decoding frame.")
                break

            # Write processed frame to the output video
            output.write(annotated_frame)

            # Stream the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        finalize_video(output, video_name)
        cancel_processing_route()

    cap.release()
    output.release()

def finalize_video(output, video_name):
    # Ensure that the output video is finalized and closed properly
    output.release()

    # Save the output video in the database after processing is done
    conn = sqlite3.connect('sharkVideoStore.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE sharkVideoStore SET output_video = ? WHERE input_video = ?", (video_name, video_name))
    conn.commit()
    conn.close()

def create_thumbnail(video_path, video_name):
    # Capture the first frame of the video to use as a thumbnail
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        thumbnail_path = os.path.join('static/thumbnails', video_name.replace('.mp4', '.jpg'))
        cv2.imwrite(thumbnail_path, frame)  # Save the thumbnail
    cap.release()

# Route to show generated videos for the user
@app.route('/videos/<email>')
def user_videos(email):
    conn = sqlite3.connect('sharkVideoStore.db')
    cursor = conn.cursor()
    cursor.execute("SELECT input_video, output_video, original_filename, timestamp FROM sharkVideoStore WHERE email = ?", (email,))
    videos = cursor.fetchall()
    conn.close()

    return render_template('videos.html', email=email, videos=videos)

if __name__ == '__main__':
    init_db()  # Initialize the database
    app.run(debug=False, threaded=True)
