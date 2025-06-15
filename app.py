import cv2
import math
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import threading

app = Flask(__name__)

latest_detection = {
    'class_name': 'N/A',
    'confidence': 0.0
}
# Lock to ensure thread-safe access to the latest_detection variable
detection_lock = threading.Lock()

try:
    model = YOLO('./runs/detect/train/weights/best.pt')
except Exception as e:
    print(f'Error loading custom model: {e}')
    print('Falling back to pretrained YOLOv11n model.')
    model = YOLO('./yolo11n.pt')

class_names = model.names

def generate_frames():
    """
    Generator function to capture frames from the webcam,
    process them for object detection, and yield them as JPEG images.
    """
    # Open the default camera (index 0)
    global latest_detection
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a new frame from the webcam
        success, frame = camera.read()
        if not success:
            print("Error: Failed to grab frame.")
            break
        else:
            # --- Perform Inference ---
            # The model processes the frame and returns detection results.
            results = model(frame, stream=True, verbose=False) # verbose=False to reduce console output
            top_detection = None

            # --- Process and Draw Detections ---
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    # Get confidence and class ID
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    if top_detection is None or confidence > top_detection['confidence']:
                        top_detection = {
                            'class_name': class_names[int(box.cls[0])],
                            'confidence': confidence,
                            'box': box.xyxy[0]
                        }
            
            # if any object was detected, draw it and update global state
            if top_detection:
                with detection_lock:
                    latest_detection['class_name'] = top_detection['class_name']
                    latest_detection['confidence'] = top_detection['confidence']
                
                # draw bounding box for the top detection
                x1, y1, x2, y2 = map(int, top_detection['box'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Prepare and display the label
                label = f'{top_detection["class_name"]}: {top_detection["confidence"]:.2f}'
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y1_label = max(y1, h + 10)
                cv2.rectangle(frame, (x1, y1_label - h - 10), (x1 + w, y1_label), (255, 0, 255), cv2.FILLED)
                cv2.putText(frame, label, (x1 + 3, y1_label - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # If no object is detected, reset global state
                with detection_lock:
                    latest_detection['class_name'] = 'N/A'
                    latest_detection['confidence'] = 0.0

            # --- Encode the Frame ---
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue # Skip frame if encoding fails
            
            frame_bytes = buffer.tobytes()

            # --- Yield the Frame ---
            # Yield the frame in the format required for multipart streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the camera when the loop is broken
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_detection_data')
def latest_detection_data():
    """Endpoint to get the latest detection data as JSON."""
    with detection_lock:
        return jsonify(latest_detection)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)