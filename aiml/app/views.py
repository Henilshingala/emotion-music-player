"""
Main views for the emotion scanner application.
Handles camera interface and emotion detection.
"""

from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.contrib import messages
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from .emotion_detector import EmotionDetector
from .models import EmotionSong, Playlist, MusicSession, SongPlay

# Initialize the emotion detector
detector = EmotionDetector()

def camera_view(request):
    """Main camera view for emotion detection"""
    return render(request, 'app/camera.html')

@csrf_exempt
@require_http_methods(["POST"])
def detect_emotion(request):
    """API endpoint for emotion detection"""
    try:
        data = json.loads(request.body)
        image_data = data.get('image')
        
        if not image_data:
            return JsonResponse({'error': 'No image data provided'}, status=400)
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format (BGR)
        image_rgb = np.array(image)
        if len(image_rgb.shape) == 3:
            image_array = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            image_array = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
        
        # Detect emotion
        result = detector.detect_emotion(image_array)
        
        if result:
            emotion = result.get('dominant_emotion')
            if not emotion:
                emotion = result.get('emotion')
            
            # Get confidence from emotions dict
            emotions_dict = result.get('emotions', {})
            if emotion and emotion in emotions_dict:
                confidence = emotions_dict[emotion]
            else:
                confidence = result.get('confidence', 0)
            
            return JsonResponse({
                'emotion': emotion,
                'confidence': confidence,
                'success': True,
                'redirect_url': f'/music/{emotion}/'  # Add redirect URL for music player
            })
        else:
            return JsonResponse({
                'error': 'No face detected in the image',
                'success': False
            }, status=400)
            
    except Exception as e:
        return JsonResponse({
            'error': f'Error processing image: {str(e)}',
            'success': False
        }, status=500)

def music_player_view(request, emotion):
    """Music player view for detected emotion"""
    try:
        # Validate emotion
        valid_emotions = [choice[0] for choice in EmotionSong.EMOTION_CHOICES]
        if emotion not in valid_emotions:
            raise Http404("Invalid emotion")
        
        # Get or create playlist for this emotion
        playlist, created = Playlist.objects.get_or_create(
            emotion=emotion,
            defaults={
                'name': f'{emotion.title()} Playlist',
                'description': f'Music for {emotion} emotions',
                'is_active': True
            }
        )
        
        # Get songs for this emotion
        songs = playlist.get_songs()
        
        if not songs.exists():
            messages.warning(request, f'No songs available for {emotion} emotion. Please contact admin to add songs.')
        
        # Create music session
        session = MusicSession.objects.create(
            emotion=emotion,
            playlist=playlist,
            user_ip=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', '')
        )
        
        context = {
            'emotion': emotion,
            'emotion_display': emotion.title(),
            'playlist': playlist,
            'songs': songs,
            'session_id': session.id,
            'total_songs': songs.count(),
        }
        
        return render(request, 'app/music_player.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading music player: {str(e)}')
        return redirect('camera')

@csrf_exempt
@require_http_methods(["POST"])
def track_song_play(request):
    """API endpoint to track song plays"""
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        song_id = data.get('song_id')
        action = data.get('action')  # 'start', 'pause', 'complete'
        
        if not all([session_id, song_id, action]):
            return JsonResponse({'error': 'Missing required parameters'}, status=400)
        
        session = get_object_or_404(MusicSession, id=session_id)
        song = get_object_or_404(EmotionSong, id=song_id)
        
        if action == 'start':
            # Create new song play record
            song_play = SongPlay.objects.create(
                session=session,
                song=song
            )
            song.increment_play_count()
            
            return JsonResponse({
                'success': True,
                'play_id': song_play.id,
                'message': 'Song play started'
            })
            
        elif action == 'complete':
            # Mark song as completed
            play_id = data.get('play_id')
            if play_id:
                try:
                    song_play = SongPlay.objects.get(id=play_id, session=session)
                    song_play.completed = True
                    song_play.save()
                except SongPlay.DoesNotExist:
                    pass
            
            return JsonResponse({
                'success': True,
                'message': 'Song play completed'
            })
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        return JsonResponse({
            'error': f'Error tracking song play: {str(e)}',
            'success': False
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def end_music_session(request):
    """API endpoint to end music session"""
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        
        if not session_id:
            return JsonResponse({'error': 'Missing session_id'}, status=400)
        
        session = get_object_or_404(MusicSession, id=session_id)
        session.session_end = timezone.now()
        session.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Music session ended'
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'Error ending session: {str(e)}',
            'success': False
        }, status=500)

def get_playlist_api(request, emotion):
    """API endpoint to get playlist data"""
    try:
        valid_emotions = [choice[0] for choice in EmotionSong.EMOTION_CHOICES]
        if emotion not in valid_emotions:
            return JsonResponse({'error': 'Invalid emotion'}, status=400)
        
        playlist = get_object_or_404(Playlist, emotion=emotion, is_active=True)
        songs = playlist.get_songs()
        
        songs_data = []
        for song in songs:
            songs_data.append({
                'id': song.id,
                'title': song.title,
                'artist': song.artist,
                'album': song.album or '',
                'duration': str(song.duration) if song.duration else None,
                'file_url': song.song_file.url if song.song_file else None,
                'cover_url': song.cover_image.url if song.cover_image else None,
                'play_count': song.play_count
            })
        
        return JsonResponse({
            'success': True,
            'playlist': {
                'id': playlist.id,
                'name': playlist.name,
                'description': playlist.description,
                'emotion': playlist.emotion,
                'cover_url': playlist.cover_image.url if playlist.cover_image else None,
                'song_count': playlist.song_count,
                'total_duration': playlist.total_duration
            },
            'songs': songs_data
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'Error getting playlist: {str(e)}',
            'success': False
        }, status=500)

def get_client_ip(request):
    """Helper function to get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

@csrf_exempt
@require_http_methods(["POST"])
def train_emotions_api(request):
    """API endpoint for training emotions with multiple images"""
    try:
        data = json.loads(request.body)
        training_data = data.get('training_data')
        
        if not training_data:
            return JsonResponse({'error': 'No training data provided'}, status=400)
        
        # Process training data - keep as base64 strings for the detector
        processed_data = {}
        total_samples = 0
        
        for emotion, images in training_data.items():
            if images:  # Only process emotions with images
                processed_data[emotion] = images  # Keep as base64 strings
                total_samples += len(images)
        
        if total_samples < 5:
            return JsonResponse({
                'error': 'Need at least 5 training samples total',
                'success': False
            }, status=400)
        
        # Train the model with processed data
        result = detector.train_custom_model(processed_data)
        
        if result.get('success'):
            return JsonResponse({
                'success': True,
                'message': result.get('message', f'Model trained successfully with {total_samples} samples'),
                'accuracy': result.get('accuracy', 0.85),
                'samples_used': result.get('samples_used', total_samples)
            })
        else:
            return JsonResponse({
                'error': result.get('error', 'Training failed'),
                'success': False
            }, status=500)
            
    except Exception as e:
        return JsonResponse({
            'error': f'Error training model: {str(e)}',
            'success': False
        }, status=500)
