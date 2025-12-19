# Emotion-Based Music Player 🎵🤖

An intelligent web application that uses facial expression recognition to detect user emotions and automatically recommends/plays music tailored to the detected mood.

## 🚀 Features

- **Real-time Emotion Detection**: Uses webcam streaming to analyze facial expressions.
- **Support for Multiple Emotions**: Detects Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral states.
- **Custom AI Model**: Includes a Voting Classifier (RandomForest + SVM) for robust emotion recognition.
- **Dynamic Playlist Recommendation**: Automatically suggests songs based on the user's current mood.
- **Heuristic Backup**: Features an advanced heuristic detection system if the custom model is not loaded.
- **Admin Dashboard**: Comprehensive interface for managing songs, playlists, and users.

## 🛠️ Technology Stack

- **Backend**: Python 3.11, Django 5.2.6
- **Computer Vision**: OpenCV
- **Machine Learning**: Scikit-learn (RandomForest, SVM), NumPy
- **Image Processing**: Pillow (PIL)
- **Frontend**: HTML5, Vanilla CSS, JavaScript

## � Installation & Setup

### 1. Prerequisite
Ensure you have **Python 3.11** installed on your system.

### 2. Environment Setup
Navigate to the project directory:
```bash
# Clone the repository
git clone https://github.com/Henilshingala/emotion-music-player.git
cd emotion-music-player
```

Create a virtual environment:
```powershell
python -m venv venv
```

Activate the virtual environment:
- **Windows**: `.\venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup
Apply the migrations to set up the SQLite database:
```bash
python manage.py migrate
```

(Optional) Populate the database with initial song data:
```bash
python add_songs_to_db.py
```

### 5. Run the Application
Start the Django development server:
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` in your web browser to start the emotion detection.

## 📸 Output Screenshots

Here are some screenshots showcasing the application's interface and functionality:

| | |
|---|---|
| ![Landing Page](https://raw.githubusercontent.com/Henilshingala/Output-images/master/emotion-music-player/1.png) |<br> ![Emotion Detection](https://raw.githubusercontent.com/Henilshingala/Output-images/master/emotion-music-player/2.png) |<br>
| ![Music Player Interface](https://raw.githubusercontent.com/Henilshingala/Output-images/master/emotion-music-player/3.png) |<br> ![Playlist View](https://raw.githubusercontent.com/Henilshingala/Output-images/master/emotion-music-player/4.png) |<br>
| ![Model Training Interface](https://raw.githubusercontent.com/Henilshingala/Output-images/master/emotion-music-player/5.png) |<br> ![Analysis Results](https://raw.githubusercontent.com/Henilshingala/Output-images/master/emotion-music-player/6.png) |<br>
| ![Song Management](https://raw.githubusercontent.com/Henilshingala/Output-images/master/emotion-music-player/7.png) |<br> ![User Preferences](https://raw.githubusercontent.com/Henilshingala/Output-images/master/emotion-music-player/8.png) |<br>

---
*Reference for Output Images:* [Henilshingala/Output-images](https://github.com/Henilshingala/Output-images/tree/master/emotion-music-player)
