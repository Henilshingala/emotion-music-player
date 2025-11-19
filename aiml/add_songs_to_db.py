#!/usr/bin/env python
import os
import sys
import django
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'emotion_player.settings')
django.setup()

from app.models import EmotionSong, Playlist
from django.core.files import File

def add_songs_from_folder():
    """Add all downloaded songs to the database based on filename"""
    
    songs_folder = Path("downloaded_songs")
    if not songs_folder.exists():
        print("❌ Downloaded songs folder not found!")
        return
    
    # Emotion mapping
    emotion_mapping = {
        'happy': 'happy',
        'sad': 'sad', 
        'angry': 'angry',
        'fear': 'fear',
        'surprise': 'surprise',
        'disgust': 'disgust',
        'neutral': 'neutral'
    }
    
    # Artist mapping for better metadata
    artist_mapping = {
        'happy': 'Happy Vibes Artist',
        'sad': 'Melancholy Artist',
        'angry': 'Intense Artist', 
        'fear': 'Calming Artist',
        'surprise': 'Dynamic Artist',
        'disgust': 'Cleansing Artist',
        'neutral': 'Balanced Artist'
    }
    
    added_count = 0
    
    # Create playlists first
    print("🎵 Setting up playlists...")
    playlist_data = [
        ('happy', 'Happy Vibes', 'Uplifting and joyful music to enhance your happiness'),
        ('sad', 'Melancholy Moods', 'Soothing and reflective music for contemplative moments'),
        ('angry', 'Intense Energy', 'Powerful music to channel your intensity'),
        ('fear', 'Calming Comfort', 'Gentle and reassuring music to ease anxiety'),
        ('surprise', 'Dynamic Discovery', 'Exciting and unexpected musical journeys'),
        ('disgust', 'Cleansing Clarity', 'Music to refresh and reset your mindset'),
        ('neutral', 'Balanced Harmony', 'Peaceful and balanced music for any mood'),
    ]
    
    for emotion_key, name, description in playlist_data:
        playlist, created = Playlist.objects.get_or_create(
            emotion=emotion_key,
            defaults={
                'name': name,
                'description': description,
                'is_active': True
            }
        )
        if created:
            print(f"✅ Created playlist: {name}")
    
    # Process each song file
    print("\n🎵 Adding songs to database...")
    
    for file_path in songs_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ['.mp3', '.mp4', '.webm', '.wav']:
            # Extract emotion from filename (e.g., "Happy.mp3" -> "happy")
            emotion_name = file_path.stem.lower()
            
            if emotion_name in emotion_mapping:
                emotion = emotion_mapping[emotion_name]
                
                # Check if song already exists
                existing_song = EmotionSong.objects.filter(
                    emotion=emotion,
                    title__icontains=emotion_name.title()
                ).first()
                
                if existing_song:
                    print(f"⚠️  Song already exists for {emotion_name}: {existing_song.title}")
                    continue
                
                try:
                    # Create the song entry
                    song = EmotionSong.objects.create(
                        title=f"{emotion_name.title()} Vibes",
                        artist=artist_mapping.get(emotion, "Unknown Artist"),
                        album=f"{emotion_name.title()} Collection",
                        emotion=emotion,
                        is_active=True,
                        play_count=0
                    )
                    
                    # Add the file
                    with open(file_path, 'rb') as f:
                        song.song_file.save(
                            f"{emotion_name}_{file_path.suffix}",
                            File(f),
                            save=True
                        )
                    
                    print(f"✅ Added {emotion_name.title()} song: {song.title}")
                    added_count += 1
                    
                except Exception as e:
                    print(f"❌ Error adding {emotion_name} song: {str(e)}")
            else:
                print(f"⚠️  Unknown emotion in filename: {file_path.name}")
    
    print(f"\n🎉 Successfully added {added_count} songs to the database!")
    
    # Show summary
    total_songs = EmotionSong.objects.filter(is_active=True).count()
    print(f"📊 Total active songs in database: {total_songs}")
    
    print("\n📋 Songs by emotion:")
    for emotion in emotion_mapping.keys():
        count = EmotionSong.objects.filter(emotion=emotion, is_active=True).count()
        print(f"   {emotion.title()}: {count} songs")

if __name__ == "__main__":
    print("🎵 Adding downloaded songs to database...")
    add_songs_from_folder()
    print("\n✅ Done! You can now test the music player pages.")
