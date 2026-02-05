import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Paths
DATASET_DIR = BASE_DIR / "data" / "dataset"
ENCODINGS_DIR = BASE_DIR / "data" / "encodings"
DATABASE_DIR = BASE_DIR / "data" / "database"

# Create directories if they don't exist
for directory in [DATASET_DIR, ENCODINGS_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_PATH = DATABASE_DIR / "attendance.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Face recognition settings
FACE_RECOGNITION_MODEL = "hog"  # "hog" for CPU, "cnn" for GPU (more accurate but slower)
TOLERANCE = 0.6  # Lower is more strict (0.6 is default)
FRAME_SKIP = 2  # Process every nth frame for performance
MIN_FACE_SIZE = 50  # Minimum face size in pixels

# Attendance settings
ATTENDANCE_TIME_WINDOW = 30  # Minutes for considering attendance
MAX_RECOGNITION_ATTEMPTS = 3  # Number of frames to confirm recognition
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score

# Audio settings
ENABLE_AUDIO = True
AUDIO_RATE = 150  # Speech rate

# Streamlit settings
STREAMLIT_TITLE = "AI Attendance System"
STREAMLIT_PORT = 8501