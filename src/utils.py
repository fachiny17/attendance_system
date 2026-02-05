import os
import json
from datetime import datetime, date
from pathlib import Path
import cv2

def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_date_string():
    """Get current date string"""
    return date.today().strftime("%Y-%m-%d")

def resize_image(image, max_width=800):
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    if width > max_width:
        ratio = max_width / width
        new_width = max_width
        new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height))
    
    return image

def validate_image_file(filepath):
    """Validate if file is an image"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_ext = Path(filepath).suffix.lower()
    return file_ext in valid_extensions and os.path.exists(filepath)

def calculate_confidence(distance):
    """Convert face distance to confidence percentage"""
    # distance is between 0 and 1, where 0 is perfect match
    confidence = max(0, min(100, (1 - distance) * 100))
    return round(confidence, 1)

def format_time_delta(delta):
    """Format time delta to human readable string"""
    seconds = delta.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{int(minutes)} minutes"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)} hours, {int(minutes)} minutes"