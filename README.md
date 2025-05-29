# SurfSafe AI

SurfSafe AI is a Flask-based web application that uses a YOLOv8 custom-trained model to detect sharks in uploaded videos. The system stores the videos and detection results in a local SQLite database.

## 📁 Project Structure

├── .vscode/ # VS Code settings (optional)
├── static/ # Folder to store uploaded video files
├── templates/ # HTML templates for rendering in Flask
├── app.py # Main Flask application file
├── requirements.txt # Python dependencies
├── sharkVideoStore.db # SQLite database for video's metadata storage
├── yolov8_custom_shark_detection_V1.pt # YOLOv8 trained shark detection model


## 🛠️ Setup Instructions

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/rashmin29/surfsafe-ai.git
cd surfsafe-ai

### 2. Create the virtual environment and install the dependencies 

python -m venv virtualenv

Windows:
.\virtualenv\Scripts\activate

Mac:
source virtualenv/bin/activate

Install the depedencies:
pip install -r requirements.txt


### 3. Run the flask app

python app.py 



