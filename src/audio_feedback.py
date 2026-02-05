import pyttsx3
import threading
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENABLE_AUDIO, AUDIO_RATE

class AudioFeedback:
    def __init__(self):
        self.engine = None
        if ENABLE_AUDIO:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', AUDIO_RATE)
                # Configure voice (optional)
                voices = self.engine.getProperty('voices')
                if voices:
                    self.engine.setProperty('voice', voices[0].id)  # First available voice
            except Exception as e:
                print(f"Audio engine initialization failed: {e}")
                self.engine = None
    
    def speak(self, text):
        """Speak text in a separate thread to avoid blocking"""
        if not self.engine:
            print(f"[AUDIO]: {text}")
            return
        
        def _speak():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Error in speech synthesis: {e}")
        
        # Run in thread to avoid blocking
        thread = threading.Thread(target=_speak)
        thread.daemon = True
        thread.start()
    
    def attendance_marked(self, student_name):
        """Audio feedback for marked attendance"""
        message = f"{student_name}, your attendance has been marked for today."
        self.speak(message)
    
    def unknown_person(self):
        """Audio feedback for unknown person"""
        message = "Unknown person detected. Please contact administrator."
        self.speak(message)
    
    def already_marked(self, student_name):
        """Audio feedback when attendance already marked"""
        message = f"{student_name}, your attendance was already marked today."
        self.speak(message)
    
    def error_occurred(self):
        """Audio feedback for errors"""
        message = "An error occurred. Please try again."
        self.speak(message)

# Global audio instance
audio = AudioFeedback()