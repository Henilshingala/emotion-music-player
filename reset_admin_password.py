#!/usr/bin/env python
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'emotion_player.settings')
django.setup()

from django.contrib.auth.models import User

def reset_admin_password():
    """Reset admin password to a simple one for testing"""
    try:
        admin_user = User.objects.get(username='admin')
        
        # Set a simple password for testing
        new_password = 'admin123'
        admin_user.set_password(new_password)
        admin_user.save()
        
        print("✅ Admin password reset successfully!")
        print(f"Username: admin")
        print(f"Password: {new_password}")
        print("\nYou can now login to /admin/ with these credentials.")
        print("Remember to change the password after logging in!")
        
    except User.DoesNotExist:
        print("❌ Admin user not found!")
        print("Please create a superuser first with: python manage.py createsuperuser")

if __name__ == "__main__":
    reset_admin_password()
