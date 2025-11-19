from django.urls import path
from . import views
from . import training_views

urlpatterns = [
    # Main views
    path('', views.camera_view, name='camera'),
    path('detect-emotion/', views.detect_emotion, name='detect_emotion'),
    
    # Training views
    path('train/', training_views.training_view, name='training'),
    path('submit-training-image/', training_views.submit_training_image, name='submit_training_image'),
    path('train-model/', training_views.train_model_with_images, name='train_model'),
    path('training-stats/', training_views.training_statistics, name='training_stats'),
    path('train-emotions/', views.train_emotions_api, name='train_emotions_api'),
    
    # Music player views
    path('music/<str:emotion>/', views.music_player_view, name='music_player'),
    path('api/playlist/<str:emotion>/', views.get_playlist_api, name='get_playlist_api'),
    path('api/track-play/', views.track_song_play, name='track_song_play'),
    path('api/end-session/', views.end_music_session, name='end_music_session'),
]


