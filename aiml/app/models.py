from django.db import models
from django.contrib.auth.models import User

class EmotionSong(models.Model):
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('angry', 'Angry'),
        ('fear', 'Fear'),
        ('surprise', 'Surprise'),
        ('disgust', 'Disgust'),
        ('neutral', 'Neutral'),
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
        help_text="Song title",
        default="Untitled"
    )
    
    artist = models.CharField(
        max_length=200,
        help_text="Artist name",
        default="Unknown Artist"
    )
    
    album = models.CharField(
        max_length=200,
        help_text="Album name (optional)",
        blank=True,
        null=True
    )
    
    duration = models.DurationField(
        help_text="Song duration (optional)",
        blank=True,
        null=True
    )
    
    cover_image = models.ImageField(
        upload_to='album_covers/',
        help_text="Album cover image (optional)",
        blank=True,
        null=True
    )
    
    order = models.PositiveIntegerField(
        default=0,
        help_text="Order in playlist (0 = first)"
    )
    
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this song is active in playlists"
    )
    
    play_count = models.PositiveIntegerField(
        default=0,
        help_text="Number of times this song has been played"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['emotion', 'order', 'title']
        verbose_name = "Emotion Song"
        verbose_name_plural = "Emotion Songs"
    
    def __str__(self):
        return f"{self.get_emotion_display()}: {self.title} by {self.artist}"
    
    def increment_play_count(self):
        self.play_count += 1
        self.save(update_fields=['play_count'])
    
    @property
    def file_size(self):
        """Get file size in MB"""
        try:
            return round(self.song_file.size / (1024 * 1024), 2)
        except:
            return 0


class Playlist(models.Model):
    """Model for emotion-based playlists"""
    emotion = models.CharField(
        max_length=20,
        choices=EmotionSong.EMOTION_CHOICES,
        unique=True,
        help_text="Emotion for this playlist"
    )
    
    name = models.CharField(
        max_length=200,
        help_text="Playlist name"
    )
    
    description = models.TextField(
        help_text="Playlist description",
        blank=True
    )
    
    cover_image = models.ImageField(
        upload_to='playlist_covers/',
        help_text="Playlist cover image (optional)",
        blank=True,
        null=True
    )
    
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this playlist is active"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['emotion']
        verbose_name = "Emotion Playlist"
        verbose_name_plural = "Emotion Playlists"
    
    def __str__(self):
        return f"{self.get_emotion_display()} Playlist: {self.name}"
    
    def get_songs(self):
        """Get all active songs for this emotion"""
        return EmotionSong.objects.filter(
            emotion=self.emotion,
            is_active=True
        ).order_by('order', 'title')
    
    @property
    def song_count(self):
        return self.get_songs().count()
    
    @property
    def total_duration(self):
        """Calculate total playlist duration"""
        songs = self.get_songs().exclude(duration__isnull=True)
        if songs:
            total = sum([song.duration.total_seconds() for song in songs], 0)
            return int(total)
        return 0


class MusicSession(models.Model):
    """Model to track music listening sessions"""
    emotion = models.CharField(
        max_length=20,
        choices=EmotionSong.EMOTION_CHOICES,
        help_text="Detected emotion that triggered this session"
    )
    
    playlist = models.ForeignKey(
        Playlist,
        on_delete=models.CASCADE,
        help_text="Playlist played in this session"
    )
    
    songs_played = models.ManyToManyField(
        EmotionSong,
        through='SongPlay',
        help_text="Songs played in this session"
    )
    
    session_start = models.DateTimeField(auto_now_add=True)
    session_end = models.DateTimeField(null=True, blank=True)
    
    user_ip = models.GenericIPAddressField(
        help_text="User's IP address",
        null=True,
        blank=True
    )
    
    user_agent = models.TextField(
        help_text="User's browser information",
        blank=True
    )
    
    class Meta:
        ordering = ['-session_start']
        verbose_name = "Music Session"
        verbose_name_plural = "Music Sessions"
    
    def __str__(self):
        return f"Session: {self.get_emotion_display()} - {self.session_start.strftime('%Y-%m-%d %H:%M')}"


class SongPlay(models.Model):
    """Model to track individual song plays within sessions"""
    session = models.ForeignKey(
        MusicSession,
        on_delete=models.CASCADE
    )
    
    song = models.ForeignKey(
        EmotionSong,
        on_delete=models.CASCADE
    )
    
    play_start = models.DateTimeField(auto_now_add=True)
    play_duration = models.DurationField(
        help_text="How long the song was played",
        null=True,
        blank=True
    )
    
    completed = models.BooleanField(
        default=False,
        help_text="Whether the song was played to completion"
    )
    
    class Meta:
        ordering = ['-play_start']
        verbose_name = "Song Play"
        verbose_name_plural = "Song Plays"
    
    def __str__(self):
        return f"{self.song.title} - {self.play_start.strftime('%H:%M:%S')}"


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


