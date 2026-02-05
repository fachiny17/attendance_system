import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime, date
from collections import defaultdict
import time

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ENCODINGS_DIR, FACE_RECOGNITION_MODEL, 
    TOLERANCE, FRAME_SKIP, MIN_FACE_SIZE,
    MAX_RECOGNITION_ATTEMPTS, CONFIDENCE_THRESHOLD
)
from src.database_handler import db_handler
from src.audio_feedback import audio

class FaceRecognizer:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()
        
        # Tracking for consistent recognition
        self.recognition_history = defaultdict(list)
        self.attendance_marked_today = set()
        self.load_todays_attendance()
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
    
    def load_encodings(self):
        """Load pre-trained facial encodings"""
        encodings_file = ENCODINGS_DIR / "encodings.pkl"
        
        if encodings_file.exists():
            with open(encodings_file, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
            print(f"Loaded {len(self.known_names)} encodings.")
        else:
            print("No encodings file found. Please enroll students first.")
            self.known_encodings = []
            self.known_names = []
    
    def load_todays_attendance(self):
        """Load today's already marked attendance"""
        today = date.today()
        # This would normally come from database
        # For now, we'll track in memory
        pass
    
    def recognize_faces(self, frame):
        """
        Recognize faces in a frame and mark attendance
        Returns: List of (name, confidence, location) for each face
        """
        self.frame_count += 1
        
        # Skip frames for performance (process every FRAME_SKIP-th frame)
        if self.frame_count % FRAME_SKIP != 0:
            return []
        
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(
            rgb_frame, model=FACE_RECOGNITION_MODEL
        )
        
        # Filter out small faces
        face_locations = [
            loc for loc in face_locations 
            if (loc[2] - loc[0]) > MIN_FACE_SIZE and (loc[1] - loc[3]) > MIN_FACE_SIZE
        ]
        
        if not face_locations or not self.known_encodings:
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            return []
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations
        )
        
        recognized_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_encodings, face_encoding, tolerance=TOLERANCE
            )
            
            name = "Unknown"
            confidence = 0
            
            if True in matches:
                # Find the best match
                face_distances = face_recognition.face_distance(
                    self.known_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = max(0, min(100, int((1 - face_distances[best_match_index]) * 100)))
            
            recognized_faces.append({
                "name": name,
                "confidence": confidence,
                "location": (top, right, bottom, left),
                "timestamp": datetime.now()
            })
            
            # Track recognition for consistency
            self.track_recognition(name, confidence)
            
            # Mark attendance if confident
            if name != "Unknown" and confidence >= CONFIDENCE_THRESHOLD * 100:
                self.mark_attendance_if_confident(name, confidence)
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        return recognized_faces
    
    def track_recognition(self, name, confidence):
        """Track recognition history for consistency"""
        current_time = datetime.now()
        
        # Clean old entries (older than 10 seconds)
        self.recognition_history[name] = [
            (t, c) for t, c in self.recognition_history[name]
            if (current_time - t).seconds < 10
        ]
        
        # Add new recognition
        self.recognition_history[name].append((current_time, confidence))
    
    def mark_attendance_if_confident(self, name, confidence):
        """Mark attendance if we have consistent recognition"""
        recognitions = self.recognition_history[name]
        
        if len(recognitions) >= MAX_RECOGNITION_ATTEMPTS:
            # Check if we have enough high-confidence recognitions
            high_confidence = [c for _, c in recognitions if c >= CONFIDENCE_THRESHOLD * 100]
            
            if len(high_confidence) >= MAX_RECOGNITION_ATTEMPTS:
                # Find student ID from name (simplified - in real app, use database)
                students = db_handler.get_all_students()
                student = next((s for s in students if s.name == name), None)
                
                if student and student.student_id not in self.attendance_marked_today:
                    # Mark attendance in database
                    success = db_handler.mark_attendance(
                        student.student_id, name, confidence
                    )
                    
                    if success:
                        self.attendance_marked_today.add(student.student_id)
                        audio.attendance_marked(name)
                        print(f"✓ Attendance marked for {name} (Confidence: {confidence}%)")
                        
                        # Clear history for this student
                        self.recognition_history[name] = []
                    else:
                        audio.already_marked(name)
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.processing_times:
            return {
                "avg_processing_time": 0,
                "estimated_fps": 0,
                "faces_known": len(self.known_names)
            }

        avg_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1 / avg_time if avg_time > 0 else 0

        return {
            "avg_processing_time": avg_time,
            "estimated_fps": fps,
            "faces_known": len(self.known_names)
        }

def run_recognition_system():
    """Main function to run the face recognition system"""
    recognizer = FaceRecognizer()
    
    if not recognizer.known_encodings:
        print("No encodings found. Please enroll students first.")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("\n" + "="*50)
    print("FACE RECOGNITION SYSTEM ACTIVE")
    print("="*50)
    print("Press 'q' to quit")
    print("Press 'r' to reload encodings")
    print("="*50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Recognize faces
        recognized_faces = recognizer.recognize_faces(frame)
        
        # Draw results on frame
        display_frame = frame.copy()
        
        # Draw header
        cv2.putText(display_frame, "AI Attendance System", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display_frame, "Press 'q' to quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw recognition results
        for face in recognized_faces:
            name = face["name"]
            confidence = face["confidence"]
            top, right, bottom, left = face["location"]
            
            # Choose color based on recognition
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                label = "Unknown"
            else:
                if confidence >= 70:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence >= 50:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                
                label = f"{name} ({confidence}%)"
            
            # Draw rectangle and label
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(display_frame, label, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display performance stats
        stats = recognizer.get_performance_stats()
        stats_text = f"FPS: {stats['estimated_fps']:.1f} | Known faces: {stats['faces_known']}"
        cv2.putText(display_frame, stats_text, (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display attendance status
        attendance_count = len(recognizer.attendance_marked_today)
        attendance_text = f"Today's attendance: {attendance_count}"
        cv2.putText(display_frame, attendance_text, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow("Attendance System", display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nShutting down...")
            break
        elif key == ord('r'):
            print("Reloading encodings...")
            recognizer.load_encodings()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    print("\n" + "="*50)
    print("SESSION SUMMARY")
    print("="*50)
    print(f"Total attendance marked today: {len(recognizer.attendance_marked_today)}")
    print(f"Average processing time: {stats['avg_processing_time']*1000:.1f}ms")
    print("="*50)

if __name__ == "__main__":
    run_recognition_system()