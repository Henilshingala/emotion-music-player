import os
import yt_dlp

# Emotion-wise YouTube links
songs = {
    "Happy": "https://www.youtube.com/watch?v=3Bn9p3-hUm8",
    "Sad": "https://www.youtube.com/watch?v=bzSTpdcs-EI",
    "Angry": "https://www.youtube.com/watch?v=vs1IDdap3X4",
    "Fear": "https://www.youtube.com/watch?v=Mm21SSgUHe8",
    "Surprise": "https://www.youtube.com/watch?v=XllMnMHdiLE",
    "Disgust": "https://www.youtube.com/watch?v=XLJCtZK0x5M",
    "Neutral": "https://www.youtube.com/watch?v=mQiiw7uRngk"
}

# Folder to save
SAVE_PATH = "downloaded_songs"
os.makedirs(SAVE_PATH, exist_ok=True)

# yt-dlp options
def download_song(emotion, link):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(SAVE_PATH, f"{emotion}.%(ext)s"),  # save as emotion.mp3
        "postprocessors": [
            {  
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

# Download loop
for emotion, url in songs.items():
    try:
        print(f"\n⬇️ Downloading {emotion} song...")
        download_song(emotion, url)
        print(f"✅ Saved as {emotion}.mp3")
    except Exception as e:
        print(f"❌ Failed {emotion}: {e}")

print("\n🎶 All songs downloaded as MP3 inside 'downloaded_songs' folder!")
