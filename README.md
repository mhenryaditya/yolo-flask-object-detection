# 🤖 Real-Time Object Detection with YOLO and Flask

## 📄 Introduction

This project is a web application that performs real-time object detection on a live webcam feed. It uses a **Flask** backend to run a **YOLO (You Only Look Once)** model on video frames captured by the server's webcam 📹. The processed video stream, complete with bounding boxes and class labels, is then sent to a web interface where it can be viewed from any browser on the same network.

This implementation uses a **server-side processing** approach, meaning all the heavy lifting (AI inference) is done by the server, and clients simply view the result.

**🎯 Core Technologies:**
* **Flask:** A lightweight Python web framework used for the backend server.
* **OpenCV:** A computer vision library used to capture and process video from the webcam.
* **Ultralytics YOLO:** The Python package used to load and run the YOLO model. *Note: While the prompt mentioned YOLOv11, this project uses the standard Ultralytics library, which officially supports versions like YOLOv8, YOLOv9, and YOLOv10.*

---

## ⚙️ Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. 📁 Project Structure

Ensure your project files are organized as follows:

```
/your_project_folder
├── 📄 app.py                 # The main Flask application
├── 📑 yolo11n.pt             # Yolo 11 version nano  
├── 📂 runs/
│   └── 📂 detect/
│       └── 📂 train/
│           └── 📂 weights/
│               └── 🏆 best.pt  # Your custom trained YOLO model
└── 📂 templates/
    └── 📄 index.html         # The HTML frontend
```

### 2. 📦 Create a Virtual Environment (Recommended)

It's good practice to create a virtual environment to manage project dependencies.

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. ⬇️ Install Dependencies

Install all the required Python libraries.

```bash
pip install flask ultralytics opencv-python
```

### 4. 🏆 Place Your Model

Make sure your trained YOLO model (e.g., `best.pt`) is located in the correct directory as specified in the project structure. The `app.py` script is configured to look for it there. If it's not found, the app will fall back to using a pretrained `yolov11n.pt` model, which it will download automatically.

### 5. 🚀 Run the Server

Start the Flask development server from your terminal.

```bash
python app.py
```

The terminal will show that the server is running, usually on `http://0.0.0.0:5000`. This means it's accessible from other devices on your local network.

### 6. 🖥️ View the Application

Open a web browser and navigate to `http://127.0.0.1:5000` (if on the same machine) or `http://<your-server-ip-address>:5000` (if viewing from another device).

---

## 📖 Resources for Learning Flask

Flask is a powerful and easy-to-learn framework. If you want to dive deeper, the official documentation is the best place to start.

* **Official Flask Tutorial:** [🔗 Flask Quickstart Guide](https://flask.palletsprojects.com/en/3.0.x/quickstart/)

This guide provides a comprehensive introduction to all the core concepts of Flask, from routing and templates to handling requests.
