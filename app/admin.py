from django.contrib import admin
from .models import TrainingImage, EmotionDetectionLog

@admin.register(TrainingImage)
class TrainingImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'labeled_emotion', 'detected_emotion', 'is_correct', 'wrong_attempts', 'needs_review', 'created_at']
    list_filter = ['labeled_emotion', 'detected_emotion', 'is_correct', 'needs_review', 'created_at']
    search_fields = ['labeled_emotion', 'detected_emotion']
    ordering = ['-created_at']
    readonly_fields = ['created_at', 'updated_at', 'wrong_attempts']
    
    fieldsets = (
        ('Image Information', {
            'fields': ('image', 'labeled_emotion', 'user')
        }),
        ('Detection Results', {
            'fields': ('detected_emotion', 'confidence_score', 'is_correct', 'wrong_attempts')
        }),
        ('Review Status', {
            'fields': ('needs_review',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.order_by('-needs_review', '-created_at')


@admin.register(EmotionDetectionLog)
class EmotionDetectionLogAdmin(admin.ModelAdmin):
    list_display = ['image', 'detected_emotion', 'confidence_score', 'detection_timestamp']
    list_filter = ['detected_emotion', 'detection_timestamp']
    search_fields = ['image__labeled_emotion', 'detected_emotion']
    ordering = ['-detection_timestamp']
    readonly_fields = ['detection_timestamp']
    
    fieldsets = (
        ('Detection Information', {
            'fields': ('image', 'detected_emotion', 'confidence_score')
        }),
        ('Timestamp', {
            'fields': ('detection_timestamp',)
        }),
    )


