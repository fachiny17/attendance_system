import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
import sqlite3
import os
from PIL import Image
import io
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import time
import threading
import subprocess
import pickle
import sys
import signal

from config import DATABASE_PATH, STREAMLIT_TITLE, DATASET_DIR, ENCODINGS_DIR
from src.database_handler import db_handler
from src.enrollment import StudentEnrollment
from src.audio_feedback import audio

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_TITLE,
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .camera-frame {
        border: 3px solid #1E88E5;
        border-radius: 10px;
        padding: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Global variable for recognition process
recognition_process = None

class StreamlitFaceRecognizer:
    """Face recognizer adapted for Streamlit"""
    def __init__(self):
        import face_recognition
        import pickle
        
        self.face_recognition = face_recognition
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()
        
        # Tracking for consistent recognition
        self.recognition_history = {}
        self.attendance_marked_today = set()
        self.load_todays_attendance()
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        self.last_update = time.time()
        
        # Streamlit state
        self.frame_placeholder = None
        self.stats_placeholder = None
        self.stop_recognition = False
        
    def load_encodings(self):
        """Load pre-trained facial encodings"""
        encodings_file = ENCODINGS_DIR / "encodings.pkl"
        
        if encodings_file.exists():
            with open(encodings_file, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
            st.success(f"✅ Loaded {len(self.known_names)} facial encodings")
            return True
        else:
            st.error("❌ No encodings file found. Please enroll students first.")
            return False
    
    def load_todays_attendance(self):
        """Load today's already marked attendance"""
        today = date.today()
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT student_id FROM attendance WHERE date = ?", (today,))
        records = cursor.fetchall()
        self.attendance_marked_today = {record[0] for record in records}
        conn.close()
    
    def recognize_frame(self, frame):
        """Recognize faces in a single frame"""
        self.frame_count += 1
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = self.face_recognition.face_locations(rgb_frame, model="hog")
        
        if not face_locations or not self.known_encodings:
            return frame, []
        
        # Get face encodings
        face_encodings = self.face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = self.face_recognition.compare_faces(
                self.known_encodings, face_encoding, tolerance=0.6
            )
            
            name = "Unknown"
            confidence = 0
            
            if True in matches:
                # Find the best match
                face_distances = self.face_recognition.face_distance(
                    self.known_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = max(0, min(100, int((1 - face_distances[best_match_index]) * 100)))
            
            recognized_faces.append({
                "name": name,
                "confidence": confidence,
                "location": (top, right, bottom, left)
            })
            
            # Track recognition
            self.track_recognition(name, confidence)
            
            # Mark attendance if confident
            if name != "Unknown" and confidence >= 70:
                self.mark_attendance_if_confident(name, confidence)
        
        # Draw results on frame
        display_frame = frame.copy()
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
        
        return display_frame, recognized_faces
    
    def track_recognition(self, name, confidence):
        """Track recognition history for consistency"""
        current_time = time.time()
        
        if name not in self.recognition_history:
            self.recognition_history[name] = []
        
        # Clean old entries (older than 10 seconds)
        self.recognition_history[name] = [
            (t, c) for t, c in self.recognition_history[name]
            if (current_time - t) < 10
        ]
        
        # Add new recognition
        self.recognition_history[name].append((current_time, confidence))
    
    def mark_attendance_if_confident(self, name, confidence):
        """Mark attendance if we have consistent recognition"""
        recognitions = self.recognition_history.get(name, [])
        
        if len(recognitions) >= 3:  # Require 3 consecutive recognitions
            # Check if we have enough high-confidence recognitions
            high_confidence = [c for _, c in recognitions if c >= 70]
            
            if len(high_confidence) >= 3:
                # Find student ID from name
                students = db_handler.get_all_students()
                student = next((s for s in students if s.name == name), None)
                
                if student and student.student_id not in self.attendance_marked_today:
                    # Mark attendance in database
                    success = db_handler.mark_attendance(
                        student.student_id, name, confidence
                    )
                    
                    if success:
                        self.attendance_marked_today.add(student.student_id)
                        
                        # Audio feedback
                        try:
                            audio.attendance_marked(name)
                        except:
                            pass
                        
                        # Streamlit success message
                        st.success(f"✅ Attendance marked for {name}!")
                        
                        # Clear history for this student
                        self.recognition_history[name] = []
                    else:
                        try:
                            audio.already_marked(name)
                        except:
                            pass

def delete_student_permanently(student_id):
    """Permanently delete student from database and remove all related files"""
    try:
        # Delete from database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Delete attendance records first (foreign key constraint handling)
        cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        
        # Delete student
        cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        
        conn.commit()
        conn.close()
        
        # Delete student images directory
        student_dir = DATASET_DIR / student_id
        if student_dir.exists():
            shutil.rmtree(student_dir)
            print(f"Deleted student directory: {student_dir}")
        
        # Delete encodings file to trigger retraining
        encodings_file = ENCODINGS_DIR / "encodings.pkl"
        if encodings_file.exists():
            os.remove(encodings_file)
            print(f"Deleted encodings file to trigger retraining")
        
        return True
    except Exception as e:
        print(f"Error deleting student: {e}")
        return False

def run_recognition_streamlit():
    """Run face recognition in Streamlit"""
    st.title("🎥 Real-Time Face Recognition")
    
    # Check if encodings exist
    encodings_file = ENCODINGS_DIR / "encodings.pkl"
    if not encodings_file.exists():
        st.error("❌ No facial encodings found. Please enroll students first.")
        st.info("Go to **Student Management** → **Add Student** to enroll students with images.")
        return
    
    # Initialize recognizer
    recognizer = StreamlitFaceRecognizer()
    
    if not recognizer.known_encodings:
        st.error("❌ Failed to load facial encodings.")
        return
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎬 Start Recognition", type="primary", use_container_width=True):
            st.session_state.recognition_active = True
            st.rerun()
    
    with col2:
        if st.button("⏸️ Pause Recognition", type="secondary", use_container_width=True):
            if 'recognition_active' in st.session_state:
                st.session_state.recognition_active = False
            st.rerun()
    
    with col3:
        if st.button("🛑 Stop Recognition", type="secondary", use_container_width=True):
            if 'recognition_active' in st.session_state:
                st.session_state.recognition_active = False
            if 'recognition_thread' in st.session_state:
                st.session_state.recognition_thread = None
            st.rerun()
    
    st.markdown("---")
    
    # Status display
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.metric("Facial Encodings", len(recognizer.known_names))
    
    with status_col2:
        today_count = len(recognizer.attendance_marked_today)
        st.metric("Today's Attendance", today_count)
    
    with status_col3:
        total_students = len(db_handler.get_all_students())
        st.metric("Total Students", total_students)
    
    st.markdown("---")
    
    # Camera preview and recognition
    if st.session_state.get('recognition_active', False):
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.success("🎯 Recognition ACTIVE - Looking for faces...")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create placeholders
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        log_placeholder = st.empty()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Make sure it's connected and not in use by another application.")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Recognition loop
        try:
            while st.session_state.get('recognition_active', False):
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Failed to capture frame from camera.")
                    break
                
                # Recognize faces
                display_frame, faces = recognizer.recognize_frame(frame)
                
                # Convert to RGB for Streamlit
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(display_frame_rgb, channels="RGB", 
                                       caption="Live Camera Feed with Face Recognition")
                
                # Display statistics
                with stats_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Faces Detected", len(faces))
                    with col2:
                        recognized = sum(1 for f in faces if f['name'] != 'Unknown')
                        st.metric("Recognized", recognized)
                    with col3:
                        fps = 1 / (time.time() - recognizer.last_update) if recognizer.last_update else 0
                        recognizer.last_update = time.time()
                        st.metric("FPS", f"{fps:.1f}")
                
                # Display recognition log
                if faces:
                    log_data = []
                    for face in faces:
                        status = "✅ Recognized" if face['name'] != 'Unknown' else "❌ Unknown"
                        log_data.append({
                            "Status": status,
                            "Name": face['name'],
                            "Confidence": f"{face['confidence']}%",
                            "Time": datetime.now().strftime("%H:%M:%S")
                        })
                    
                    if log_data:
                        with log_placeholder.container():
                            st.subheader("Recognition Log")
                            log_df = pd.DataFrame(log_data)
                            st.dataframe(log_df, use_container_width=True, hide_index=True)
                
                # Small delay to prevent high CPU usage
                time.sleep(0.1)
        
        except Exception as e:
            st.error(f"❌ Error during recognition: {e}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            if st.session_state.get('recognition_active', False):
                st.session_state.recognition_active = False
            
            st.info("🛑 Recognition stopped.")
    
    else:
        # Show preview when not active
        st.info("👆 Click **Start Recognition** to begin face recognition and automatic attendance marking.")
        
        # Show sample preview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### How it works:")
            st.markdown("""
            1. Student stands in front of camera
            2. System detects and recognizes face
            3. If recognized with high confidence:
               - Attendance is automatically marked
               - Audio confirmation is played
            4. Unknown faces are highlighted in red
            5. All attendance is logged in real-time
            """)
        
        with col2:
            st.markdown("### Tips for best results:")
            st.markdown("""
            - ✅ Good lighting conditions
            - ✅ Face clearly visible
            - ✅ No obstructions (masks, sunglasses)
            - ✅ Multiple training images per student
            - ✅ Camera at eye level
            """)
        
        # Show recent attendance
        st.markdown("---")
        st.subheader("📊 Recent Attendance")
        
        today_records = db_handler.get_todays_attendance()
        if today_records:
            recent_data = []
            for record in today_records[-10:]:  # Last 10 records
                recent_data.append({
                    "Student": record.recognized_name or record.student_id,
                    "Time": record.time.strftime("%H:%M:%S"),
                    "Confidence": f"{record.confidence or 0}%"
                })
            
            df_recent = pd.DataFrame(recent_data)
            st.dataframe(df_recent, use_container_width=True, hide_index=True)
        else:
            st.info("No attendance marked today yet.")

def run_external_recognition():
    """Run the external recognizer.py script"""
    st.info("🚀 Launching external recognition system...")
    
    try:
        # Run the recognizer script
        process = subprocess.Popen(
            [sys.executable, "src/recognizer.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        st.session_state.recognition_process = process
        
        # Show output
        output_placeholder = st.empty()
        
        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_placeholder.text(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        if return_code == 0:
            st.success("✅ Recognition system completed successfully!")
        else:
            st.error(f"❌ Recognition system exited with code: {return_code}")
            
            # Show error output
            stderr_output = process.stderr.read()
            if stderr_output:
                st.code(stderr_output, language="bash")
    
    except Exception as e:
        st.error(f"❌ Error starting recognition system: {e}")

def stop_external_recognition():
    """Stop the external recognizer.py script"""
    if 'recognition_process' in st.session_state:
        process = st.session_state.recognition_process
        if process and process.poll() is None:  # Process is still running
            # Terminate the process
            process.terminate()
            try:
                process.wait(timeout=5)  # Wait for process to terminate
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if not terminated
            
            st.session_state.recognition_process = None
            st.success("✅ Recognition system stopped.")
        else:
            st.info("ℹ️ No active recognition process found.")
    else:
        st.info("ℹ️ No active recognition process found.")

def main():
    # Header
    st.markdown(f"<h1 class='main-header'>{STREAMLIT_TITLE}</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'recognition_active' not in st.session_state:
        st.session_state.recognition_active = False
    if 'recognition_thread' not in st.session_state:
        st.session_state.recognition_thread = None
    if 'recognition_process' not in st.session_state:
        st.session_state.recognition_process = None
    
    # Sidebar navigation - ADDED "Face Recognition" option
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Dashboard",
        "Face Recognition",  # NEW PAGE
        "Today's Attendance",
        "Attendance Reports",
        "Student Management",
        "Manual Entry",
        "System Status"
    ])
    
    # Dashboard Page
    if page == "Dashboard":
        show_dashboard()
    
    # Face Recognition Page - NEW
    elif page == "Face Recognition":
        show_face_recognition()
    
    # Today's Attendance Page
    elif page == "Today's Attendance":
        show_todays_attendance()
    
    # Attendance Reports Page
    elif page == "Attendance Reports":
        show_attendance_reports()
    
    # Student Management Page
    elif page == "Student Management":
        show_student_management()
    
    # Manual Entry Page
    elif page == "Manual Entry":
        show_manual_entry()
    
    # System Status Page
    elif page == "System Status":
        show_system_status()

def show_face_recognition():
    """Show face recognition page with multiple options"""
    st.title("🤖 Face Recognition Options")
    
    # Option selection
    recognition_mode = st.radio(
        "Choose recognition mode:",
        [
            "Streamlit Web Interface",
            "External Terminal Application"
        ],
        horizontal=True
    )
    
    st.markdown("---")
    
    if recognition_mode == "Streamlit Web Interface":
        st.markdown("### 🎥 Streamlit Web Interface")
        st.markdown("""
        **Features:**
        - Real-time face recognition directly in your browser
        - Live camera feed with face detection
        - Automatic attendance marking
        - Real-time statistics and logs
        - No terminal required
        
        **Requirements:**
        - Webcam connected to your computer
        - Modern browser with camera permissions
        - Good lighting conditions
        """)
        
        run_recognition_streamlit()
    
    else:  # External Terminal Application
        st.markdown("### 💻 External Terminal Application")
        st.markdown("""
        **Features:**
        - Full-featured recognition system
        - Terminal-based interface
        - Better performance for large datasets
        - Audio feedback (text-to-speech)
        - Detailed logging
        
        **How to use:**
        1. Click "Launch Recognition System" below
        2. A new terminal window will open
        3. System will start recognizing faces
        4. Press 'q' in the terminal to quit
        5. Or use the "Stop Recognition" button below
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Launch Recognition System", type="primary", use_container_width=True):
                # Run in a thread to avoid blocking
                import threading
                
                def run_external():
                    run_external_recognition()
                
                thread = threading.Thread(target=run_external)
                thread.daemon = True
                thread.start()
                
                st.success("✅ Recognition system launched! Check your terminal.")
        
        with col2:
            if st.button("🛑 Stop Recognition System", type="secondary", use_container_width=True):
                stop_external_recognition()
        
        # Show status if process is running
        if st.session_state.get('recognition_process'):
            process = st.session_state.recognition_process
            if process and process.poll() is None:
                st.success("✅ External recognition system is running")
            else:
                st.info("ℹ️ External recognition system is not running")

def show_dashboard():
    """Display main dashboard with metrics and charts"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get today's date
    today = date.today()
    
    # Calculate metrics
    total_students = len(db_handler.get_all_students())
    todays_attendance = len(db_handler.get_todays_attendance())
    attendance_rate = (todays_attendance / total_students * 100) if total_students > 0 else 0
    
    # Get department count
    students = db_handler.get_all_students()
    departments = set(s.department for s in students if s.department)
    dept_count = len(departments)
    
    # Display metrics
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Students</div>
        </div>
        """.format(total_students), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Today's Attendance</div>
        </div>
        """.format(todays_attendance), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}%</div>
            <div class="metric-label">Attendance Rate</div>
        </div>
        """.format(attendance_rate), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Departments</div>
        </div>
        """.format(dept_count), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick actions with recognition button
    st.subheader("Quick Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🎥 Start Face Recognition", use_container_width=True, type="primary"):
            st.session_state.page = "Face Recognition"
            st.rerun()
    
    with action_col2:
        if st.button("📥 Export Today's Report", use_container_width=True):
            export_todays_report()
    
    with action_col3:
        if st.button("➕ Add Student", use_container_width=True):
            st.session_state.page = "Student Management"
            st.rerun()
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        # Recent attendance timeline
        st.subheader("Recent Attendance")
        today_records = db_handler.get_todays_attendance()
        if today_records:
            # Convert to DataFrame for easier manipulation
            today_data = []
            for record in today_records[-20:]:  # Last 20 records
                today_data.append({
                    'Student': record.recognized_name or record.student_id,
                    'Time': record.time,
                    'Confidence': record.confidence or 0
                })
            
            today_df = pd.DataFrame(today_data)
            today_df['Time'] = pd.to_datetime(today_df['Time'].astype(str))
            today_df = today_df.sort_values('Time')
            
            # Create scatter plot
            fig = px.scatter(today_df, x='Time', y='Student',
                            color='Confidence',
                            title='Recent Attendance Timeline',
                            color_continuous_scale='Viridis')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No attendance marked today yet.")
    
    with col2:
        # Department attendance
        st.subheader("Department Overview")
        students = db_handler.get_all_students()
        if students:
            # Count students by department
            dept_counts = {}
            for student in students:
                dept = student.department or 'Unknown'
                dept_counts[dept] = dept_counts.get(dept, 0) + 1
            
            dept_df = pd.DataFrame(list(dept_counts.items()), columns=['Department', 'Count'])
            
            fig = px.pie(dept_df, values='Count', names='Department',
                        title='Students by Department',
                        hole=0.3)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No students enrolled yet.")

# ... (Keep all the other functions: show_todays_attendance, show_attendance_reports,
# show_student_management, show_manual_entry, show_system_status, export_todays_report)
# These functions remain the same as in your original code

def show_todays_attendance():
    """Display today's attendance records"""
    st.header("Today's Attendance")
    
    # Date selector
    selected_date = st.date_input("Select Date", date.today())
    
    # Get daily report
    report_df = db_handler.get_daily_report(selected_date)
    
    if report_df.empty:
        st.info(f"No attendance records for {selected_date}")
    else:
        # Display statistics
        total = len(report_df)
        present = len(report_df[report_df['status'] == 'Present'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", total)
        col2.metric("Present", present, f"{present/total*100:.1f}%" if total > 0 else "0%")
        
        # Department breakdown
        if 'department' in report_df.columns:
            dept_stats = report_df.groupby('department')['status'].apply(
                lambda x: (x == 'Present').sum() / len(x) * 100
            ).reset_index()
            dept_stats.columns = ['Department', 'Attendance %']
            col3.metric(
                "Top Department",
                dept_stats.loc[dept_stats['Attendance %'].idxmax(), 'Department'] if not dept_stats.empty else "N/A",
                f"{dept_stats['Attendance %'].max():.1f}%" if not dept_stats.empty else "N/A"
            )
        
        # Filter options
        st.subheader("Attendance Records")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_status = st.selectbox("Filter by Status", ["All", "Present", "Absent"])
        with col2:
            filter_dept = st.selectbox("Filter by Department", 
                                      ["All"] + list(report_df['department'].dropna().unique()))
        
        # Apply filters
        filtered_df = report_df.copy()
        if filter_status != "All":
            filtered_df = filtered_df[filtered_df['status'] == filter_status]
        if filter_dept != "All":
            filtered_df = filtered_df[filtered_df['department'] == filter_dept]
        
        # Display table
        st.dataframe(
            filtered_df[['student_id', 'name', 'department', 'status', 'time', 'confidence']],
            use_container_width=True,
            column_config={
                "student_id": "Student ID",
                "name": "Student Name",
                "department": "Department",
                "status": "Status",
                "time": "Time",
                "confidence": "Confidence (%)"
            }
        )
        
        # Export option
        if st.button("Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"attendance_{selected_date}.csv",
                mime="text/csv"
            )

def show_attendance_reports():
    """Display attendance reports and analytics"""
    st.header("Attendance Reports")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", date.today() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", date.today())
    
    if start_date > end_date:
        st.error("Start date must be before end date")
        return
    
    # Generate report
    report_df = db_handler.get_attendance_report(start_date, end_date)
    
    if report_df.empty:
        st.info("No attendance data for the selected period.")
        return
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    total_records = len(report_df)
    unique_students = report_df['Student ID'].nunique()
    avg_daily = total_records / report_df['Date'].nunique() if report_df['Date'].nunique() > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", total_records)
    col2.metric("Unique Students", unique_students)
    col3.metric("Avg Daily Attendance", f"{avg_daily:.1f}")
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["Daily Trend", "Student Analysis", "Raw Data"])
    
    with tab1:
        # Daily attendance trend
        daily_counts = report_df.groupby('Date').size().reset_index(name='Count')
        fig = px.bar(daily_counts, x='Date', y='Count', 
                    title='Daily Attendance Count',
                    color='Count',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Student-wise attendance
        student_counts = report_df['Name'].value_counts().reset_index()
        student_counts.columns = ['Student', 'Count']
        student_counts = student_counts.head(20)  # Top 20 students
        
        fig = px.bar(student_counts, x='Student', y='Count',
                    title='Top 20 Students by Attendance Count',
                    color='Count',
                    color_continuous_scale='Viridis')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Raw data table
        st.dataframe(report_df, use_container_width=True)
        
        # Export option
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="Download Full Report as CSV",
            data=csv,
            file_name=f"attendance_report_{start_date}_to_{end_date}.csv",
            mime="text/csv"
        )

def show_student_management():
    """Manage student records"""
    st.header("Student Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["View Students", "Add Student", "Edit Student", "Manage Images"])
    
    with tab1:
        # View all students
        students = db_handler.get_all_students()
        
        if not students:
            st.info("No students enrolled yet.")
        else:
            # Convert to DataFrame for display
            student_data = []
            for student in students:
                student_id = student.student_id
                # Check if student has images
                student_dir = DATASET_DIR / student_id
                has_images = student_dir.exists() and any(
                    f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                    for f in os.listdir(student_dir) if student_dir.is_dir()
                )
                
                student_data.append({
                    'Student ID': student_id,
                    'Name': student.name,
                    'Department': student.department or 'Not set',
                    'Email': student.email or 'N/A',
                    'Enrolled Date': student.created_at.date(),
                    'Has Images': '✅' if has_images else '❌'
                })
            
            df = pd.DataFrame(student_data)
            st.dataframe(df, use_container_width=True)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Students", len(students))
            col2.metric("With Images", len(df[df['Has Images'] == '✅']))
            
            # Count by department
            if df['Department'].nunique() > 0:
                dept_counts = df['Department'].value_counts()
                col3.metric("Departments", dept_counts.index[0], f"{dept_counts.iloc[0]} students")
    
    with tab2:
        # Add new student
        st.subheader("Add New Student")
        
        with st.form("add_student_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                student_id = st.text_input("Student ID*")
                name = st.text_input("Full Name*")
            
            with col2:
                department = st.text_input("Department")
                email = st.text_input("Email")
            
            # Image upload section
            st.subheader("Add Student Images")
            uploaded_files = st.file_uploader(
                "Upload face images for recognition (JPG/PNG)",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Upload multiple face images from different angles"
            )
            
            submitted = st.form_submit_button("Add Student")
            
            if submitted:
                if not student_id or not name:
                    st.error("Student ID and Name are required!")
                else:
                    # Add student to database
                    success = db_handler.add_student(
                        student_id, name, department, email
                    )
                    
                    if success:
                        # Save uploaded images
                        if uploaded_files:
                            student_dir = DATASET_DIR / student_id
                            student_dir.mkdir(exist_ok=True)
                            
                            saved_count = 0
                            for i, uploaded_file in enumerate(uploaded_files):
                                # Save image
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"{student_id}_{timestamp}_{i:02d}.jpg"
                                filepath = student_dir / filename
                                
                                # Read and save image
                                image = Image.open(uploaded_file)
                                image.save(filepath)
                                saved_count += 1
                            
                            st.success(f"✓ Student {name} added successfully with {saved_count} images!")
                            
                            # Train encodings
                            enrollment = StudentEnrollment()
                            trained_count = enrollment.train_encodings()
                            if trained_count > 0:
                                st.success(f"✓ Facial encodings trained successfully! ({trained_count} encodings)")
                            else:
                                st.warning("⚠ Could not train facial encodings. Check image quality.")
                        else:
                            st.success(f"✓ Student {name} added successfully! No images uploaded.")
                        
                        st.rerun()
                    else:
                        st.error("✗ Failed to add student. Student ID might already exist.")
    
    with tab3:
        # Edit student
        st.subheader("Edit Student Information")
        
        students = db_handler.get_all_students()
        if students:
            student_options = {f"{s.student_id} - {s.name}": s.student_id for s in students}
            selected = st.selectbox("Select student to edit", list(student_options.keys()))
            
            if selected:
                student_id = student_options[selected]
                student = db_handler.get_student(student_id)
                
                if student:
                    # Create two columns for form
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Edit Information")
                        with st.form("edit_student_form"):
                            new_name = st.text_input("Name", value=student.name)
                            new_department = st.text_input("Department", value=student.department or "")
                            new_email = st.text_input("Email", value=student.email or "")
                            
                            submitted_edit = st.form_submit_button("Update Student")
                            
                            if submitted_edit:
                                success = db_handler.update_student(
                                    student_id, new_name, new_department, new_email
                                )
                                if success:
                                    st.success(f"✓ Student {student_id} updated successfully!")
                                    st.rerun()
                                else:
                                    st.error("✗ Failed to update student.")
                    
                    with col2:
                        st.subheader("Delete Student")
                        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
                        st.warning("⚠ Warning: This action cannot be undone!")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        delete_option = st.radio(
                            "Choose delete option:",
                            ["Soft Delete (Mark as inactive)", "Permanent Delete (Remove all data)"]
                        )
                        
                        with st.form("delete_student_form"):
                            confirm_input = st.text_input(f"Type '{student_id}' to confirm deletion:")
                            
                            col_del1, col_del2 = st.columns(2)
                            with col_del1:
                                submitted_delete = st.form_submit_button("Delete Student", type="secondary")
                            
                            if submitted_delete:
                                if confirm_input.strip() == student_id:
                                    if delete_option == "Soft Delete (Mark as inactive)":
                                        # Soft delete (mark inactive)
                                        deleted = db_handler.delete_student(student_id)
                                        if deleted:
                                            st.success(f"✓ Student {student_id} marked inactive (soft delete).")
                                            st.rerun()
                                        else:
                                            st.error("✗ Failed to soft delete student.")
                                    else:
                                        # Permanent delete
                                        st.markdown("<div class='danger-box'>", unsafe_allow_html=True)
                                        st.error("🚨 Are you sure? This will permanently delete ALL data for this student!")
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        
                                        # Double confirmation
                                        double_confirm = st.checkbox("I understand this will permanently delete all student data")
                                        if double_confirm:
                                            deleted = delete_student_permanently(student_id)
                                            if deleted:
                                                st.success(f"✓ Student {student_id} permanently deleted!")
                                                st.rerun()
                                            else:
                                                st.error("✗ Failed to delete student permanently.")
                                else:
                                    st.error("Confirmation mismatch: type the exact Student ID to confirm deletion.")
                else:
                    st.error("Student not found.")
        else:
            st.info("No students to edit.")
    
    with tab4:
        # Manage student images
        st.subheader("Manage Student Images")
        
        students = db_handler.get_all_students()
        if students:
            student_options = {f"{s.student_id} - {s.name}": s.student_id for s in students}
            selected = st.selectbox("Select student", list(student_options.keys()))
            
            if selected:
                student_id = student_options[selected]
                student_dir = DATASET_DIR / student_id
                
                # Display current images
                st.subheader(f"Current Images for {selected}")
                
                if student_dir.exists():
                    image_files = [f for f in os.listdir(student_dir) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    if image_files:
                        # Show image count
                        st.info(f"Found {len(image_files)} images for this student.")
                        
                        # Display images in a grid
                        cols = st.columns(3)
                        for i, image_file in enumerate(image_files):
                            with cols[i % 3]:
                                try:
                                    image_path = student_dir / image_file
                                    image = Image.open(image_path)
                                    
                                    # Create a small preview
                                    st.image(image, caption=f"Image {i+1}", use_column_width=True)
                                    
                                    # Delete button for each image
                                    if st.button(f"Delete", key=f"delete_{i}"):
                                        os.remove(image_path)
                                        st.success(f"Deleted {image_file}")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error loading image: {e}")
                    else:
                        st.warning("No images found for this student.")
                else:
                    st.warning("No image directory found for this student.")
                
                # Add new images section
                st.subheader("Add New Images")
                with st.form("add_images_form"):
                    new_uploaded_files = st.file_uploader(
                        "Upload new face images",
                        type=['jpg', 'jpeg', 'png'],
                        accept_multiple_files=True,
                        help="Upload additional face images"
                    )
                    
                    submitted_add = st.form_submit_button("Upload Images")
                    
                    if submitted_add and new_uploaded_files:
                        # Create directory if it doesn't exist
                        student_dir.mkdir(exist_ok=True)
                        
                        saved_count = 0
                        for i, uploaded_file in enumerate(new_uploaded_files):
                            # Generate unique filename
                            existing_count = len([f for f in os.listdir(student_dir) 
                                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{student_id}_{timestamp}_{existing_count + i:02d}.jpg"
                            filepath = student_dir / filename
                            
                            # Save image
                            image = Image.open(uploaded_file)
                            image.save(filepath)
                            saved_count += 1
                        
                        st.success(f"✓ {saved_count} new images uploaded!")
                        
                        # Retrain encodings
                        enrollment = StudentEnrollment()
                        trained_count = enrollment.train_encodings()
                        if trained_count > 0:
                            st.success(f"✓ Facial encodings retrained successfully!")
                        else:
                            st.warning("⚠ Could not train facial encodings. Check image quality.")
                        
                        st.rerun()

def show_manual_entry():
    """Manual attendance entry"""
    st.header("Manual Attendance Entry")
    
    # Get all students
    students = db_handler.get_all_students()
    
    if not students:
        st.info("No students enrolled. Please add students first.")
        return
    
    # Create student selection (use student_id when roll number is not available)
    student_options = {f"{getattr(s, 'roll_number', s.student_id)} - {s.name}": s.student_id for s in students}
    
    with st.form("manual_entry_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            selected_student = st.selectbox(
                "Select Student",
                options=list(student_options.keys())
            )
            
            entry_date = st.date_input("Date", date.today())
        
        with col2:
            entry_time = st.time_input("Time", datetime.now().time())
            status = st.selectbox("Status", ["Present", "Absent", "Late"])
        
        notes = st.text_area("Notes (Optional)")
        
        submitted = st.form_submit_button("Mark Attendance")
        
        if submitted:
            student_id = student_options[selected_student]
            student_name = selected_student.split(" - ")[1]
            
            # Check if already marked for today
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM attendance 
                WHERE student_id = ? AND date = ?
            """, (student_id, entry_date))
            
            exists = cursor.fetchone()[0] > 0
            
            if exists and status == "Present":
                st.warning(f"Attendance already marked for {student_name} on {entry_date}")
            else:
                # Insert manual entry
                cursor.execute("""
                    INSERT INTO attendance (student_id, date, time, status, recognized_name)
                    VALUES (?, ?, ?, ?, ?)
                """, (student_id, entry_date, entry_time, status, student_name))
                
                conn.commit()
                conn.close()
                
                st.success(f"Attendance marked for {student_name} on {entry_date}")
                st.rerun()

def show_system_status():
    """Display system status and information"""
    st.header("System Status")
    
    # System information
    st.subheader("Database Information")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get table counts
    cursor.execute("SELECT COUNT(*) FROM students")
    student_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM attendance")
    attendance_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (date.today(),))
    today_count = cursor.fetchone()[0]
    
    conn.close()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", student_count)
    col2.metric("Total Attendance Records", attendance_count)
    col3.metric("Today's Records", today_count)
    
    # Storage information
    st.subheader("Storage Information")
    
    data_dir = DATABASE_PATH.parent.parent
    total_size = 0
    
    for path, dirs, files in os.walk(data_dir):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    
    total_size_mb = total_size / (1024 * 1024)
    
    st.info(f"Total data size: {total_size_mb:.2f} MB")
    
    # Dataset information
    st.subheader("Face Recognition Dataset")
    
    dataset_stats = {
        "Total Students with Images": 0,
        "Total Images": 0
    }
    
    if DATASET_DIR.exists():
        for student_dir in DATASET_DIR.iterdir():
            if student_dir.is_dir():
                image_files = [f for f in os.listdir(student_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    dataset_stats["Total Students with Images"] += 1
                    dataset_stats["Total Images"] += len(image_files)
    
    col1, col2 = st.columns(2)
    col1.metric("Students with Images", dataset_stats["Total Students with Images"])
    col2.metric("Total Images", dataset_stats["Total Images"])
    
    # Check if encodings file exists
    encodings_file = ENCODINGS_DIR / "encodings.pkl"
    if encodings_file.exists():
        st.success("✅ Facial encodings are trained and ready")
    else:
        st.warning("⚠ Facial encodings not trained yet. Run the training from Student Management.")
    
    # Webcam status
    st.subheader("Camera Status")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            st.success("✅ Webcam is accessible")
            cap.release()
        else:
            st.warning("⚠ Webcam not accessible")
    except:
        st.warning("⚠ Could not check webcam status")
    
    # Instructions
    st.subheader("System Instructions")
    
    with st.expander("How to use the system"):
        st.markdown("""
        1. **Enroll Students**: Use the enrollment script or web interface to add students
        2. **Add Images**: Upload student face images in Student Management
        3. **Train Encodings**: System automatically trains encodings when images are added
        4. **Start Recognition**: Go to **Face Recognition** page to begin automatic attendance
        5. **Monitor**: Use this dashboard to view attendance records
        6. **Manage**: Add/remove students or make manual entries as needed
        """)
    
    with st.expander("Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        
        - **No face detected**: Ensure good lighting and camera positioning
        - **Low confidence**: Add more training images from different angles
        - **Database errors**: Check if database file exists and is accessible
        - **Camera not working**: Check if another app is using the camera
        
        **For terminal-based recognition:**
        ```bash
        python src/recognizer.py
        ```
        
        **For web-based recognition:**
        - Go to **Face Recognition** page
        - Choose "Streamlit Web Interface"
        - Click "Start Recognition"
        """)
    
    # Refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()

def export_todays_report():
    """Export today's attendance report"""
    try:
        df = db_handler.get_attendance_report(date.today(), date.today())
        
        if df.empty:
            st.warning("No attendance data for today.")
            return
        
        # Convert to CSV
        csv = df.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="📥 Download Today's Attendance",
            data=csv,
            file_name=f"attendance_{date.today()}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error generating report: {e}")

if __name__ == "__main__":
    main()