from django.core.management.base import BaseCommand
from app.models import Playlist, EmotionSong

class Command(BaseCommand):
    help = 'Create default playlists for all emotions'

    def handle(self, *args, **options):
        emotions = [
            ('happy', 'Happy Vibes', 'Uplifting and joyful music to enhance your happiness'),
            ('sad', 'Melancholy Moods', 'Soothing and reflective music for contemplative moments'),
            ('angry', 'Intense Energy', 'Powerful music to channel your intensity'),
            ('fear', 'Calming Comfort', 'Gentle and reassuring music to ease anxiety'),
            ('surprise', 'Dynamic Discovery', 'Exciting and unexpected musical journeys'),
            ('disgust', 'Cleansing Clarity', 'Music to refresh and reset your mindset'),
            ('neutral', 'Balanced Harmony', 'Peaceful and balanced music for any mood'),
        ]
        
        created_count = 0
        
        for emotion_key, name, description in emotions:
            playlist, created = Playlist.objects.get_or_create(
                emotion=emotion_key,
                defaults={
                    'name': name,
                    'description': description,
                    'is_active': True
                }
            )
            
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created playlist: {name}')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'Playlist already exists: {name}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(f'\nSetup complete! Created {created_count} new playlists.')
        )
        
        # Show summary
        total_songs = EmotionSong.objects.filter(is_active=True).count()
        if total_songs == 0:
            self.stdout.write(
                self.style.WARNING(
                    '\nNo songs found! Please add MP3 files through the admin panel:'
                    '\n1. Go to /admin/'
                    '\n2. Click on "Emotion Songs"'
                    '\n3. Add songs for each emotion'
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(f'\nFound {total_songs} active songs in the system.')
            )
