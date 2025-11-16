from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import json
import base64
import io
import os
import cv2
import numpy as np
from PIL import Image
from .models import TrainingImage, EmotionDetectionLog
from .emotion_detector import EmotionDetector


class TrainingManager:
    
    def __init__(self):
        self.emotion_detector = EmotionDetector()
    
    def save_training_image(self, image_data, emotion, user=None):
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            filename = f"training_{emotion}_{user.id if user else 'anonymous'}_{os.urandom(4).hex()}.jpg"
            image_file = ContentFile(image_bytes, name=filename)
            training_image = TrainingImage(
                image=image_file,
                labeled_emotion=emotion,
                user=user
            )
            training_image.save()
            
            return training_image
            
        except Exception as e:
            print(f"Error saving training image: {e}")
            return None
    
    def detect_emotion_on_training_image(self, training_image):
        try:
            image_path = training_image.image.path
            image_cv = cv2.imread(image_path)
            
            if image_cv is None:
                return None
            
            result = self.emotion_detector.detect_emotion(image_cv)
            
            if result:
                detected_emotion = result['dominant_emotion'].lower()
                confidence = result['emotions'][result['dominant_emotion']]
                
                EmotionDetectionLog.objects.create(
                    image=training_image,
                    detected_emotion=detected_emotion,
                    confidence_score=confidence
                )
                is_correct = detected_emotion == training_image.labeled_emotion
                
                if not is_correct:
                    training_image.mark_wrong_detection(detected_emotion, confidence)
                else:
                    training_image.detected_emotion = detected_emotion
                    training_image.confidence_score = confidence
                    training_image.is_correct = True
                    training_image.save()
                
                return {
                    'detected_emotion': detected_emotion,
                    'confidence': confidence,
                    'is_correct': is_correct
                }
            
            return None
            
        except Exception as e:
            print(f"Error detecting emotion on training image: {e}")
            return None
    
    def get_training_statistics(self):
        total_images = TrainingImage.objects.count()
        correct_detections = TrainingImage.objects.filter(is_correct=True).count()
        wrong_detections = TrainingImage.objects.filter(is_correct=False).count()
        needs_review = TrainingImage.objects.filter(needs_review=True).count()
        
        emotion_stats = {}
        for emotion in ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']:
            emotion_images = TrainingImage.objects.filter(labeled_emotion=emotion)
            emotion_stats[emotion] = {
                'total': emotion_images.count(),
                'correct': emotion_images.filter(is_correct=True).count(),
                'wrong': emotion_images.filter(is_correct=False).count()
            }
        
        return {
            'total_images': total_images,
            'correct_detections': correct_detections,
            'wrong_detections': wrong_detections,
            'needs_review': needs_review,
            'accuracy': (correct_detections / total_images * 100) if total_images > 0 else 0,
            'emotion_stats': emotion_stats
        }


def training_view(request):
    training_manager = TrainingManager()
    stats = training_manager.get_training_statistics()
    
    context = {
        'stats': stats,
        'emotions': [
            {'key': 'happy', 'name': 'Happy', 'icon': '😊'},
            {'key': 'sad', 'name': 'Sad', 'icon': '😢'},
            {'key': 'angry', 'name': 'Angry', 'icon': '😠'},
            {'key': 'fear', 'name': 'Fear', 'icon': '😨'},
            {'key': 'surprise', 'name': 'Surprise', 'icon': '😲'},
            {'key': 'disgust', 'name': 'Disgust', 'icon': '🤢'},
            {'key': 'neutral', 'name': 'Neutral', 'icon': '😐'},
        ]
    }
    
    return render(request, 'app/training.html', context)


@csrf_exempt
def submit_training_image(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            emotion = data.get('emotion')
            
            if not image_data or not emotion:
                return JsonResponse({'success': False, 'error': 'Missing image data or emotion'})
            
            user = request.user if request.user.is_authenticated else None
            training_manager = TrainingManager()
            training_image = training_manager.save_training_image(image_data, emotion, user)
            
            if training_image:
                detection_result = training_manager.detect_emotion_on_training_image(training_image)
                
                return JsonResponse({
                    'success': True,
                    'image_id': training_image.id,
                    'detection_result': detection_result,
                    'message': f'Image saved successfully for {emotion} emotion'
                })
            else:
                return JsonResponse({'success': False, 'error': 'Failed to save image'})
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@csrf_exempt
def train_model_with_images(request):
    if request.method == 'POST':
        try:
            training_manager = TrainingManager()
            training_images = TrainingImage.objects.all()
            
            if training_images.count() < 10:
                return JsonResponse({
                    'success': False, 
                    'error': f'Need at least 10 training images. Currently have {training_images.count()}'
                })
            
            training_data = {}
            for image in training_images:
                emotion = image.labeled_emotion
                if emotion not in training_data:
                    training_data[emotion] = []
                
                try:
                    image_path = image.image.path
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        training_data[emotion].append(image_b64)
                except Exception as e:
                    print(f"Error reading image {image.id}: {e}")
                    continue
            result = training_manager.emotion_detector.train_model(training_data)
            
            if result['success']:
                return JsonResponse({
                    'success': True,
                    'accuracy': result['accuracy'],
                    'samples_used': result['samples_used'],
                    'message': f'Model trained successfully with {result["accuracy"]:.1f}% accuracy'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': result['error']
                })
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


def training_statistics(request):
    training_manager = TrainingManager()
    stats = training_manager.get_training_statistics()
    
    return JsonResponse(stats)


