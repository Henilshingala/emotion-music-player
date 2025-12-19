from django.db import models
from django.contrib.auth.models import User

class EmotionSong(models.Model):
    EMOTION_CHOICES = [
        ('Happy', 'Happy'),
        ('Sad', 'Sad'),
        ('Angry', 'Angry'),
        ('Fear', 'Fear'),
        ('Surprise', 'Surprise'),
        ('Disgust', 'Disgust'),
        ('Neutral', 'Neutral'),
    ]
    
    emotion = models.CharField(
        max_length=20,
        choices=EMOTION_CHOICES,
        help_text="Select the emotion this song represents"
    )
    
    song_file = models.FileField(
        upload_to='songs/',
        help_text="Upload an MP3 file for this emotion"
    )
    
    title = models.CharField(
        max_length=200,
        help_text="Song title (optional)",
        blank=True
    )
    
    artist = models.CharField(
        max_length=200,
        help_text="Artist name (optional)",
        blank=True
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['emotion', 'title']
        verbose_name = "Emotion Song"
        verbose_name_plural = "Emotion Songs"
    
    def __str__(self):
        if self.title and self.artist:
            return f"{self.emotion}: {self.title} by {self.artist}"
        elif self.title:
            return f"{self.emotion}: {self.title}"
        else:
            return f"{self.emotion}: {self.song_file.name}"


class TrainingImage(models.Model):
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('angry', 'Angry'),
        ('fear', 'Fear'),
        ('surprise', 'Surprise'),
        ('disgust', 'Disgust'),
        ('neutral', 'Neutral'),
    ]
    
    image = models.ImageField(
        upload_to='training_images/',
        help_text="Training image for emotion detection"
    )
    
    labeled_emotion = models.CharField(
        max_length=20,
        choices=EMOTION_CHOICES,
        help_text="The emotion this image represents"
    )
    
    detected_emotion = models.CharField(
        max_length=20,
        choices=EMOTION_CHOICES,
        blank=True,
        null=True,
        help_text="The emotion detected by the system"
    )
    
    confidence_score = models.FloatField(
        default=0.0,
        help_text="Confidence score of the detection"
    )
    
    is_correct = models.BooleanField(
        default=True,
        help_text="Whether the detection was correct"
    )
    
    wrong_attempts = models.IntegerField(
        default=0,
        help_text="Number of times this image was detected incorrectly"
    )
    
    needs_review = models.BooleanField(
        default=False,
        help_text="Whether this image needs admin review"
    )
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        help_text="User who submitted this image"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Training Image"
        verbose_name_plural = "Training Images"
    
    def __str__(self):
        return f"Training Image: {self.labeled_emotion} (ID: {self.id})"
    
    def mark_wrong_detection(self, detected_emotion, confidence):
        self.detected_emotion = detected_emotion
        self.confidence_score = confidence
        self.is_correct = False
        self.wrong_attempts += 1
        
        if self.wrong_attempts >= 3:
            self.needs_review = True
        
        self.save()


class EmotionDetectionLog(models.Model):
    """Model to log emotion detection attempts for analysis"""
    image = models.ForeignKey(
        TrainingImage,
        on_delete=models.CASCADE,
        related_name='detection_logs'
    )
    
    detected_emotion = models.CharField(
        max_length=20,
        choices=TrainingImage.EMOTION_CHOICES
    )
    
    confidence_score = models.FloatField()
    
    detection_timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-detection_timestamp']
        verbose_name = "Emotion Detection Log"
        verbose_name_plural = "Emotion Detection Logs"
    
    def __str__(self):
        return f"Detection: {self.detected_emotion} ({self.confidence_score:.2f})"


