# Emotion-Based Music Player - Django AI Application

## 📋 Project Overview
**Emotion Music Player** is an innovative **Django web application** that uses AI and computer vision to detect user emotions through facial expressions and recommend music accordingly. This cutting-edge app combines machine learning, emotion recognition, and music curation to create a personalized listening experience.

## 🛠️ Technology Stack
- **Backend Framework**: Django 5.2.6
- **Computer Vision**: OpenCV 4.10.0
- **Image Processing**: Pillow 10.2.0
- **Machine Learning**: scikit-learn 1.3.0
- **Numerical Computing**: NumPy 1.26.4
- **AI Model**: Custom emotion detection model
- **Database**: SQLite3
- **Python Version**: 3.11 (venv311)

## ✨ Key Features

### 1. **Emotion Detection**
- Real-time facial emotion recognition
- Webcam integration via OpenCV
- AI-powered emotion classification
- Support for multiple emotions:
  - Happy
  - Sad
  - Angry
  - Surprised
  - Neutral
  - Fear
  - Disgust

### 2. **Music Recommendation**
- Emotion-based song selection
- Curated playlists for each emotion
- Automatic music playback
- Seamless  song transitions

### 3. **AI/ML Integration**
- Custom emotion detection model (`custom_emotion_model.pkl`)
- Emotion scaler for normalization (`emotion_scaler.pkl`)
- Scikit-learn for ML operations

- Face detection and feature extraction
- Real-time prediction

### 4. **Media Management**
- Song library management
- Support for various audio formats
- Media file organization
- Playlist creation

### 5. **User Interface**
- Clean, intuitive design
- Real-time emotion display
- Music player controls
- Camera feed display

## 📁 Project Structure
```
emotion-music-player-main/
├── emotion_player/           # Main Django project
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── app/                      # Main application
│   ├── models.py            # Database models
│   ├── views.py             # View logic
│   ├── urls.py              # URL routing
│   ├── templates/           # HTML templates
│   └── static/              # CSS, JS, images
│
├── aiml/                     # AI/ML components
│   ├── emotion_detection.py # Emotion recognition
│   ├── model_training.py    # Model training scripts
│   ├── data_preprocessing.py # Data preparation
│   └── [94 files]           # ML models and data
│
├── media/                    # Uploaded media files
│   └── songs/               # Music library
│
├── venv311/                  # Python virtual environment
│   └── [12,379 files]       # Python packages
│
├── ML Models:
├── custom_emotion_model.pkl  # Trained emotion model
├── emotion_scaler.pkl        # Feature scaler
│
├── Configuration:
├── manage.py                 # Django management
├── requirements.txt          # Python dependencies
├── setup.py                  # Setup script
├── song.py                   # Song utilities
├── db.sqlite3               # Database
└── .gitattributes           # Git configuration
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher (Python 3.11 recommended)
- pip package manager
- Webcam for emotion detection
- Audio output device

### Installation Steps

1. **Navigate to project**
   ```bash
   cd emotion-music-player-main/emotion-music-player-main
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Dependencies include:**
   ```txt
   Django==5.2.6
   opencv-python==4.10.0.84
   Pillow==10.2.0
   numpy==1.26.4
   scikit-learn==1.3.0
   ```

5. **Run setup script (if available)**
   ```bash
   python setup.py install
   ```

6. **Run migrations**
   ```bash
   python manage.py migrate
   ```

7. **Create superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

8. **Add music files**
   - Place songs in `media/songs/` directory
   - Organize by emotion/mood if needed

9. **Run development server**
   ```bash
   python manage.py runserver
   ```

10. **Access application**
    ```
    http://127.0.0.1:8000
    ```

## 🎵 Music Organization

### Recommended Structure:
```
media/
└── songs/
    ├── happy/
    │   ├── song1.mp3
    │   └── song2.mp3
    ├── sad/
    │   ├── song1.mp3
    │   └── song2.mp3
    ├── energetic/
    ├── calm/
    ├── romantic/
    └── neutral/
```

### Supported Audio Formats:
- MP3
- WAV
- OGG
- AAC
- FLAC (depending on browser support)

## 🧠 AI/ML Components

### Emotion Detection Model

#### Training Process:
1. **Data Collection**: Facial expression dataset
2. **Preprocessing**: Face detection, normalization
3. **Feature Extraction**: Facial landmarks, features
4. **Model Training**: Scikit-learn classifiers
5. **Model Saving**: Pickle files (.pkl)

#### Model Files:
- **custom_emotion_model.pkl**: Trained classifier
- **emotion_scaler.pkl**: Feature normalization

### Emotion Classification:
```python
# Emotion categories detected:
emotions = [
    'happy',
    'sad',
    'angry',
    'surprised',
    'neutral',
    'fear',
    'disgust'
]
```

## 🎬 How It Works

### Application Flow:

1. **Camera Initialization**
   - OpenCV captures webcam feed
   - Real-time video processing

2. **Face Detection**
   - Detect faces in video frame
   - Extract facial features

3. **Emotion Recognition**
   - Process facial features
   - Model predicts emotion
   - Confidence score calculated

4. **Music Selection**
   - Query database for emotion-matched songs
   - Select appropriate playlist
   - Retrieve song file

5. **Music Playback**
   - Stream selected song
   - Display player controls
   - Allow user interaction

6. **Continuous Monitoring**
   - Keep detecting emotions
   - Update recommendations
   - Smooth transitions

## 🔧 Configuration

### Django Settings
Key configurations in `emotion_player/settings.py`:
- Database: SQLite3
- Media URL and ROOT
- Installed apps
- Middleware
- Template configuration

### OpenCV Configuration
Camera settings:
- Resolution
- Frame rate
- Detection frequency

## 📸 Webcam Integration

### Requirements:
- Functional webcam
- Browser camera permissions
- Adequate lighting for detection

### Browser Permissions:
The app will request:
- Camera access
- Media access

## 🎨 User Interface Features

### Main Screen:
- Live camera feed
- Detected emotion display
- Confidence percentage
- Now playing information
- Music player controls

### Player Controls:
- Play/Pause
- Skip
- Volume control
- Playlist view
- Manual emotion override

## 🔐 Privacy & Security

### Data Privacy:
- ✅ No images stored permanently
- ✅ Real-time processing only
- ✅ Local emotion detection
- ✅ No data sent to external servers

### Permissions:
- Camera access (required)
- Microphone access (not needed)
- Media storage (for music library)

## 📊 Features Breakdown

### Core Features ✅
- Real-time emotion detection
- Automatic music selection
- Music playback
- Playlist management
- User interface

### Advanced Features 🎯
- Multiple emotion support
- Confidence scoring
- Playlist creation
- History tracking
- Manual overrides

## 🎯 Use Cases

### Personal Use:
- Mood-based music discovery
- Emotional wellness
- Stress relief
- Entertainment
- Personalized playlists

### Professional Applications:
- Music therapy
- Psychological research
- Entertainment venues
- Fitness applications
- Retail ambiance

## 📈 Technical Details

### ML Model Specs:
- Algorithm: Likely SVM or Random Forest
- Input: Facial feature vectors
- Output: Emotion probabilities
- Training: scikit-learn
- Format: Pickle (.pkl)

### Performance:
- Real-time detection: ~30 FPS
- Emotion update: Every few frames
- Model inference: < 100ms
- Low latency music response

## 🚨 Troubleshooting

### Camera Not Working:
```bash
# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Test camera
python -c "import cv2; cv2.VideoCapture(0).read()"
```

### Model Not Loading:
- Verify .pkl files exist
- Check scikit-learn version compatibility
- Retrain model if needed

### Songs Not Playing:
- Check media folder structure
- Verify audio file formats
- Check browser audio support

## 🔄 Development Workflow

1. **Setup environment**
2. **Train/load emotion model**
3. **Organize music library**
4. **Run Django server**
5. **Grant camera permissions**
6. **Test emotion detection**
7. **Enjoy personalized music!**

## 📦 Dependencies Explained

```txt
Django==5.2.6           # Web framework
opencv-python==4.10.0   # Computer vision, webcam
Pillow==10.2.0          # Image processing
numpy==1.26.4           # Numerical operations
scikit-learn==1.3.0     # Machine learning
```

## 🎓 Educational Value

### Learning Topics:
- Django web development
- Computer vision with OpenCV
- Machine learning with scikit-learn
- Real-time video processing
- Emotion recognition AI
- Web-based ML applications

## 🌐 Deployment Considerations

### Heroku:
- Add Procfile
- Configure buildpacks for OpenCV
- Set environment variables

### Docker:
```dockerfile
FROM python:3.11
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# Install dependencies and run
```

### Challenges:
- Webcam access in production
- OpenCV on server environments
- Large media files
- Real-time processing load

## 💡 Enhancement Ideas

### Future Features:
- 🎵 Spotify API integration
- 📊 Emotion analytics dashboard
- 👥 Multi-user support
- 🎨 Custom themes
- 📱 Mobile app version
- 🌐 Social sharing
- 🎯 Mood history tracking
- 🤖 Improved ML model
- 🎬 Emotion-based video recommendations
- 💬 Voice control

---

**Project**: Emotion-Based Music Player
**Type**: AI-Powered Django Web Application
**Technology**: Django + OpenCV + Machine Learning
**Status**: Functional prototype

**🎵 Music That Understands Your Mood!**

*Combining the power of AI, computer vision, and music to create a truly personalized listening experience.*
