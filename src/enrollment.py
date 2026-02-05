import cv2
import os
import face_recognition
import pickle
from datetime import datetime
import sys
from pathlib import Path
import shutil

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASET_DIR, ENCODINGS_DIR
from src.database_handler import db_handler

class StudentEnrollment:
    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.encodings_dir = ENCODINGS_DIR
        # Create directories if they don't exist
        self.dataset_dir.mkdir(exist_ok=True)
        self.encodings_dir.mkdir(exist_ok=True)
    
    def capture_images(self, student_id, name, department=None, num_images=10):
        """
        Capture multiple images of a student for dataset
        """
        print(f"Enrolling student: {name} (ID: {student_id})")
        if department:
            print(f"Department: {department}")
        
        # Create student directory
        student_dir = self.dataset_dir / student_id
        student_dir.mkdir(exist_ok=True)
        
        # Check if student already exists
        existing_student = db_handler.get_student(student_id)
        if existing_student:
            print(f"⚠ Warning: Student ID {student_id} already exists.")
            overwrite = input("Do you want to overwrite? (yes/no): ").strip().lower()
            if overwrite != 'yes':
                print("Enrollment cancelled.")
                return False
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return False
        
        print("\nInstructions:")
        print("- Press SPACE to capture image")
        print("- Press ESC to quit/cancel")
        print("- Move your head slightly between captures")
        print("- Ensure good lighting and face the camera\n")
        
        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            
            # Display instructions on frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Student: {name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"ID: {student_id}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Images: {count}/{num_images}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect face in frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                # Draw rectangle around face
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(display_frame, "Face detected", (left, top-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press SPACE to capture", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(display_frame, "No face detected", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(display_frame, "Align face with camera", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow("Student Enrollment", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC to quit
                print("Enrollment cancelled.")
                break
            elif key == 32 and face_locations:  # SPACE to capture
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{student_id}_{timestamp}_{count:02d}.jpg"
                filepath = student_dir / filename
                
                cv2.imwrite(str(filepath), frame)
                print(f"✓ Captured image {count + 1}/{num_images}")
                count += 1
                
                # Show brief confirmation
                cv2.putText(display_frame, "Image Saved!", (200, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Student Enrollment", display_frame)
                cv2.waitKey(300)  # Brief pause
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        if count > 0:
            # Add student to database with department
            success = db_handler.add_student(student_id, name, department)
            
            if success:
                # Train encodings
                trained_count = self.train_encodings()
                if trained_count > 0:
                    print(f"\n✓ Enrollment successful for {name}!")
                    print(f"✓ {count} images captured")
                    print(f"✓ {trained_count} encodings trained")
                    return True
                else:
                    print("✗ Enrollment failed: Could not train encodings")
                    return False
            else:
                print("✗ Enrollment failed: Could not add student to database")
                return False
        else:
            print("✗ Enrollment failed: No images captured")
            return False
    
    def train_encodings(self):
        """
        Train facial encodings from all images in dataset
        Returns number of encodings created
        """
        print("\nTraining facial encodings...")
        
        known_encodings = []
        known_names = []
        known_ids = []  # Store student IDs for reference
        
        # Walk through dataset directory
        for student_id in os.listdir(self.dataset_dir):
            student_path = self.dataset_dir / student_id
            
            if not student_path.is_dir():
                continue
            
            # Get student info from database
            student = db_handler.get_student(student_id)
            if not student:
                print(f"⚠ Student {student_id} not found in database, skipping...")
                continue
            
            student_name = student.name
            
            # Process each image for this student
            image_count = 0
            image_files = [f for f in os.listdir(student_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"⚠ No images found for {student_name}")
                continue
            
            for image_file in image_files:
                image_path = student_path / image_file
                
                try:
                    # Load image and find face encodings
                    image = face_recognition.load_image_file(str(image_path))
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        known_encodings.append(face_encodings[0])
                        known_names.append(student_name)
                        known_ids.append(student_id)
                        image_count += 1
                    else:
                        print(f"  ⚠ No face found in {image_file}")
                except Exception as e:
                    print(f"  ⚠ Error processing {image_file}: {e}")
            
            print(f"  ✓ {student_name} ({student_id}): {image_count}/{len(image_files)} images encoded")
        
        if not known_encodings:
            print("✗ No valid encodings created. Check images and try again.")
            return 0
        
        # Save encodings to file
        encodings_data = {
            "encodings": known_encodings,
            "names": known_names,
            "ids": known_ids
        }
        
        encodings_file = self.encodings_dir / "encodings.pkl"
        with open(encodings_file, "wb") as f:
            pickle.dump(encodings_data, f)
        
        print(f"\n✓ Training complete:")
        print(f"  - Total encodings: {len(known_names)}")
        print(f"  - Saved to: {encodings_file}")
        
        return len(known_names)
    
    def add_images_to_student(self, student_id, image_files):
        """
        Add images to existing student and retrain encodings
        Returns: (success, message)
        """
        try:
            # Check if student exists
            student = db_handler.get_student(student_id)
            if not student:
                return False, f"Student {student_id} not found in database"
            
            # Create student directory
            student_dir = self.dataset_dir / student_id
            student_dir.mkdir(exist_ok=True)
            
            # Count existing images
            existing_images = [f for f in os.listdir(student_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            start_index = len(existing_images)
            
            saved_count = 0
            for i, image_file in enumerate(image_files):
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{student_id}_{timestamp}_{start_index + i:02d}.jpg"
                filepath = student_dir / filename
                
                # Save image
                image_file.save(str(filepath))
                saved_count += 1
            
            # Retrain encodings
            trained_count = self.train_encodings()
            
            return True, f"{saved_count} images added. {trained_count} encodings trained."
            
        except Exception as e:
            return False, f"Error adding images: {str(e)}"

def enroll_student_cli():
    """
    Command-line interface for student enrollment
    """
    enrollment = StudentEnrollment()
    
    print("\n" + "="*50)
    print("STUDENT ENROLLMENT SYSTEM")
    print("="*50)
    
    # Get student information
    student_id = input("Enter Student ID: ").strip()
    name = input("Enter Student Name: ").strip()
    department = input("Enter Department (optional): ").strip()
    
    if not student_id or not name:
        print("✗ Error: Student ID and Name are required.")
        return
    
    # Confirmation
    print(f"\nStudent Information:")
    print(f"  ID: {student_id}")
    print(f"  Name: {name}")
    if department:
        print(f"  Department: {department}")
    
    confirm = input("\nIs this correct? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Enrollment cancelled.")
        return
    
    # Capture images
    print(f"\nReady to capture images for {name}")
    print("Position yourself in front of the camera with good lighting.")
    input("Press Enter to start camera...")
    
    success = enrollment.capture_images(
        student_id=student_id,
        name=name,
        department=department if department else None,
        num_images=8
    )
    
    if success:
        print(f"\n" + "="*50)
        print(f"✓ ENROLLMENT SUCCESSFUL")
        print("="*50)
    else:
        print(f"\n" + "="*50)
        print(f"✗ ENROLLMENT FAILED")
        print("="*50)

# Quick enrollment function for testing
def quick_enrollment():
    """Quick enrollment for testing purposes"""
    enrollment = StudentEnrollment()
    
    test_data = [
        {"id": "CS001", "name": "John Doe", "department": "Computer Science"},
        {"id": "CS002", "name": "Jane Smith", "department": "Computer Science"},
        {"id": "EE001", "name": "Bob Johnson", "department": "Electrical Engineering"},
    ]
    
    print("Quick Enrollment Test Mode")
    for i, student in enumerate(test_data, 1):
        print(f"\n[{i}/{len(test_data)}] Enrolling {student['name']}...")
        success = enrollment.capture_images(
            student_id=student['id'],
            name=student['name'],
            department=student['department'],
            num_images=5
        )
        if success:
            print(f"✓ {student['name']} enrolled successfully")
        else:
            print(f"✗ Failed to enroll {student['name']}")
    
    print("\nQuick enrollment completed!")

if __name__ == "__main__":
    # Simple menu for enrollment
    print("\n" + "="*50)
    print("ATTENDANCE SYSTEM - ENROLLMENT MODULE")
    print("="*50)
    print("1. Enroll New Student")
    print("2. Quick Test Enrollment (for testing)")
    print("3. Train Encodings Only")
    print("="*50)
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        enroll_student_cli()
    elif choice == "2":
        quick_enrollment()
    elif choice == "3":
        enrollment = StudentEnrollment()
        count = enrollment.train_encodings()
        if count > 0:
            print(f"\n✓ Encodings trained successfully ({count} encodings)")
        else:
            print("\n✗ Failed to train encodings")
    else:
        print("Invalid choice. Exiting.")