from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import (
    EmotionSong, TrainingImage, EmotionDetectionLog, 
    Playlist, MusicSession, SongPlay
)

@admin.register(EmotionSong)
class EmotionSongAdmin(admin.ModelAdmin):
    list_display = ('emotion', 'title', 'artist', 'album', 'order', 'play_count', 'is_active', 'file_size_display', 'created_at')
    list_filter = ('emotion', 'is_active', 'created_at', 'artist')
    search_fields = ('title', 'artist', 'album')
    ordering = ('emotion', 'order', 'title')
    list_editable = ('order', 'is_active')
    
    fieldsets = (
        ('Song Information', {
            'fields': ('emotion', 'title', 'artist', 'album')
        }),
        ('Media Files', {
            'fields': ('song_file', 'cover_image', 'duration')
        }),
        ('Playlist Settings', {
            'fields': ('order', 'is_active')
        }),
        ('Statistics', {
            'fields': ('play_count',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    readonly_fields = ('created_at', 'updated_at', 'play_count')
    
    actions = ['activate_songs', 'deactivate_songs', 'reset_play_counts']
    
    def file_size_display(self, obj):
        return f"{obj.file_size} MB" if obj.file_size > 0 else "N/A"
    file_size_display.short_description = "File Size"
    
    def activate_songs(self, request, queryset):
        queryset.update(is_active=True)
        self.message_user(request, f"{queryset.count()} songs activated.")
    activate_songs.short_description = "Activate selected songs"
    
    def deactivate_songs(self, request, queryset):
        queryset.update(is_active=False)
        self.message_user(request, f"{queryset.count()} songs deactivated.")
    deactivate_songs.short_description = "Deactivate selected songs"
    
    def reset_play_counts(self, request, queryset):
        queryset.update(play_count=0)
        self.message_user(request, f"Play counts reset for {queryset.count()} songs.")
    reset_play_counts.short_description = "Reset play counts"

@admin.register(Playlist)
class PlaylistAdmin(admin.ModelAdmin):
    list_display = ('emotion', 'name', 'song_count_display', 'total_duration_display', 'is_active', 'created_at')
    list_filter = ('emotion', 'is_active', 'created_at')
    search_fields = ('name', 'description')
    ordering = ('emotion',)
    list_editable = ('is_active',)
    
    fieldsets = (
        ('Playlist Information', {
            'fields': ('emotion', 'name', 'description')
        }),
        ('Media', {
            'fields': ('cover_image',)
        }),
        ('Settings', {
            'fields': ('is_active',)
        }),
        ('Statistics', {
            'fields': ('song_count_display', 'total_duration_display'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    readonly_fields = ('created_at', 'updated_at', 'song_count_display', 'total_duration_display')
    
    def song_count_display(self, obj):
        count = obj.song_count
        if count > 0:
            url = reverse('admin:app_emotionsong_changelist') + f'?emotion={obj.emotion}'
            return format_html('<a href="{}">{} songs</a>', url, count)
        return "0 songs"
    song_count_display.short_description = "Songs"
    
    def total_duration_display(self, obj):
        duration = obj.total_duration
        if duration > 0:
            minutes = duration // 60
            seconds = duration % 60
            return f"{minutes}:{seconds:02d}"
        return "N/A"
    total_duration_display.short_description = "Duration"

@admin.register(MusicSession)
class MusicSessionAdmin(admin.ModelAdmin):
    list_display = ('emotion', 'playlist', 'session_start', 'session_end', 'songs_played_count', 'user_ip')
    list_filter = ('emotion', 'session_start', 'playlist')
    search_fields = ('user_ip', 'user_agent')
    ordering = ('-session_start',)
    date_hierarchy = 'session_start'
    
    fieldsets = (
        ('Session Information', {
            'fields': ('emotion', 'playlist')
        }),
        ('Timing', {
            'fields': ('session_start', 'session_end')
        }),
        ('User Information', {
            'fields': ('user_ip', 'user_agent'),
            'classes': ('collapse',)
        }),
    )
    
    readonly_fields = ('session_start',)
    
    def songs_played_count(self, obj):
        count = obj.songplay_set.count()
        if count > 0:
            url = reverse('admin:app_songplay_changelist') + f'?session__id={obj.id}'
            return format_html('<a href="{}">{} plays</a>', url, count)
        return "0 plays"
    songs_played_count.short_description = "Songs Played"

@admin.register(SongPlay)
class SongPlayAdmin(admin.ModelAdmin):
    list_display = ('song', 'session', 'play_start', 'play_duration', 'completed')
    list_filter = ('completed', 'play_start', 'song__emotion')
    search_fields = ('song__title', 'song__artist', 'session__emotion')
    ordering = ('-play_start',)
    date_hierarchy = 'play_start'
    
    fieldsets = (
        ('Play Information', {
            'fields': ('session', 'song')
        }),
        ('Timing', {
            'fields': ('play_start', 'play_duration', 'completed')
        }),
    )
    
    readonly_fields = ('play_start',)

@admin.register(TrainingImage)
class TrainingImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'labeled_emotion', 'detected_emotion', 'confidence_score', 'is_correct', 'needs_review', 'created_at')
    list_filter = ('labeled_emotion', 'detected_emotion', 'is_correct', 'needs_review', 'created_at')
    search_fields = ('labeled_emotion', 'detected_emotion')
    ordering = ('-created_at',)
    
    fieldsets = (
        ('Image Information', {
            'fields': ('image', 'labeled_emotion')
        }),
        ('Detection Results', {
            'fields': ('detected_emotion', 'confidence_score', 'is_correct')
        }),
        ('Review Status', {
            'fields': ('wrong_attempts', 'needs_review')
        }),
        ('User Information', {
            'fields': ('user',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    readonly_fields = ('created_at', 'updated_at')
    
    actions = ['mark_for_review', 'mark_as_correct']
    
    def mark_for_review(self, request, queryset):
        queryset.update(needs_review=True)
        self.message_user(request, f"{queryset.count()} images marked for review.")
    mark_for_review.short_description = "Mark selected images for review"
    
    def mark_as_correct(self, request, queryset):
        queryset.update(is_correct=True, needs_review=False)
        self.message_user(request, f"{queryset.count()} images marked as correct.")
    mark_as_correct.short_description = "Mark selected images as correct"

@admin.register(EmotionDetectionLog)
class EmotionDetectionLogAdmin(admin.ModelAdmin):
    list_display = ('image', 'detected_emotion', 'confidence_score', 'detection_timestamp')
    list_filter = ('detected_emotion', 'detection_timestamp')
    search_fields = ('detected_emotion', 'image__labeled_emotion')
    ordering = ('-detection_timestamp',)
    
    readonly_fields = ('detection_timestamp',)
    
    fieldsets = (
        ('Detection Information', {
            'fields': ('image', 'detected_emotion', 'confidence_score')
        }),
        ('Timestamp', {
            'fields': ('detection_timestamp',)
        }),
    )

# Customize admin site header and title
admin.site.site_header = "AI Emotion Detection & Music Admin"
admin.site.site_title = "Emotion Music Admin"
admin.site.index_title = "Welcome to Emotion Music Administration"


