"""
Main views for the emotion scanner application.
Handles camera interface and emotion detection.
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
import io
from PIL import Image
import numpy as np
import cv2
from .emotion_detector import EmotionDetector


def camera_view(request):
    return render(request, 'app/camera.html')


def advanced_emotion_detector(image_cv):
    detector = EmotionDetector()
    return detector.detect_emotion(image_cv)


@csrf_exempt
def detect_emotion(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            
            if not image_data:
                return JsonResponse({'status': 'failed', 'error': 'No image data provided'})
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            result = advanced_emotion_detector(image_cv)
            
            if result:
                emotion_scores = result['emotions']
                dominant_emotion = result['dominant_emotion']
                
                emotion_mapping = {
                    'happy': 'Happy',
                    'sad': 'Sad',
                    'angry': 'Angry',
                    'fear': 'Fear',
                    'surprise': 'Surprise',
                    'disgust': 'Disgust',
                    'neutral': 'Neutral'
                }
                
                mapped_emotion = emotion_mapping.get(dominant_emotion, dominant_emotion.capitalize())
                
                return JsonResponse({
                    'emotion': mapped_emotion,
                    'confidence': emotion_scores[dominant_emotion],
                    'all_emotions': emotion_scores
                })
            else:
                return JsonResponse({'status': 'failed', 'error': 'No face detected'})
                
        except Exception as e:
            return JsonResponse({'status': 'failed', 'error': str(e)})
    
    return JsonResponse({'status': 'failed', 'error': 'Invalid request method'})




def train_emotions_view(request):
    return render(request, 'app/train_emotions.html')


@csrf_exempt
def train_emotions_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            training_data = data.get('training_data', {})
            
            if not training_data:
                return JsonResponse({'success': False, 'error': 'No training data provided'})
            
            detector = EmotionDetector()
            result = detector.train_model(training_data)
            
            if result['success']:
                return JsonResponse({
                    'success': True, 
                    'accuracy': result['accuracy'],
                    'samples_trained': result['samples_used'],
                    'message': f'Model trained with {result["accuracy"]*100:.1f}% accuracy on {result["samples_used"]} samples'
                })
            else:
                return JsonResponse({'success': False, 'error': result['error']})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

