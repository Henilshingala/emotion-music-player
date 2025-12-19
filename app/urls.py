from django.urls import path
from . import views
from . import training_views

urlpatterns = [
    path('', views.camera_view, name='camera'),
    path('detect-emotion/', views.detect_emotion, name='detect_emotion'),
    path('train/', training_views.training_view, name='training'),
    path('submit-training-image/', training_views.submit_training_image, name='submit_training_image'),
    path('train-model/', training_views.train_model_with_images, name='train_model'),
    path('training-stats/', training_views.training_statistics, name='training_stats'),
    path('train-emotions/', views.train_emotions_api, name='train_emotions_api'),
]


