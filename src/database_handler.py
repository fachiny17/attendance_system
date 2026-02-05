from sqlalchemy import create_engine, Column, Integer, String, Date, Time, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
import pandas as pd
from config import DATABASE_URL

Base = declarative_base()

class Student(Base):
    __tablename__ = 'students'
    
    id = Column(Integer, primary_key=True)
    student_id = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    department = Column(String)
    email = Column(String)
    created_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<Student(name='{self.name}', id='{self.student_id}')>"

class Attendance(Base):
    __tablename__ = 'attendance'
    
    id = Column(Integer, primary_key=True)
    student_id = Column(String, nullable=False)
    date = Column(Date, nullable=False, default=date.today)
    time = Column(Time, nullable=False, default=lambda: datetime.now().time())
    status = Column(String, default="Present")
    recognized_name = Column(String)
    confidence = Column(Integer)
    
    def __repr__(self):
        return f"<Attendance(student='{self.student_id}', date='{self.date}', status='{self.status}')>"

class DatabaseHandler:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def add_student(self, student_id, name, department=None, email=None):
        """Add a new student to the database"""
        try:
            student = Student(
                student_id=student_id,
                name=name,
                department=department,
                email=email
            )
            self.session.add(student)
            self.session.commit()
            print(f"✓ Student {name} added successfully.")
            return True
        except Exception as e:
            print(f"✗ Error adding student: {e}")
            self.session.rollback()
            return False
    
    def update_student(self, student_id, name=None, department=None, email=None):
        """Update student information"""
        try:
            student = self.session.query(Student).filter_by(student_id=student_id).first()
            if not student:
                print(f"✗ Student {student_id} not found.")
                return False
            
            if name:
                student.name = name
            if department:
                student.department = department
            if email:
                student.email = email
            
            self.session.commit()
            print(f"✓ Student {student_id} updated successfully.")
            return True
        except Exception as e:
            print(f"✗ Error updating student: {e}")
            self.session.rollback()
            return False
    
    def mark_attendance(self, student_id, recognized_name="", confidence=0):
        """Mark attendance for a student"""
        try:
            # Check if attendance already marked today
            today = date.today()
            existing = self.session.query(Attendance).filter_by(
                student_id=student_id,
                date=today
            ).first()
            
            if existing:
                print(f"⚠ Attendance already marked for {student_id} today.")
                return False
            
            # Mark new attendance
            attendance = Attendance(
                student_id=student_id,
                recognized_name=recognized_name,
                confidence=confidence,
                status="Present"
            )
            self.session.add(attendance)
            self.session.commit()
            print(f"✓ Attendance marked for {student_id}.")
            return True
        except Exception as e:
            print(f"✗ Error marking attendance: {e}")
            self.session.rollback()
            return False
    
    def get_student(self, student_id):
        """Get student details by ID"""
        student = self.session.query(Student).filter_by(student_id=student_id).first()
        return student
    
    def get_all_students(self):
        """Get all active students"""
        students = self.session.query(Student).filter_by(is_active=True).all()
        return students
    
    def get_students_by_department(self, department):
        """Get students by department"""
        students = self.session.query(Student).filter_by(
            department=department,
            is_active=True
        ).all()
        return students
    
    def get_todays_attendance(self):
        """Get today's attendance records"""
        today = date.today()
        records = self.session.query(Attendance).filter_by(date=today).all()
        return records
    
    def get_attendance_by_date(self, target_date):
        """Get attendance for a specific date"""
        records = self.session.query(Attendance).filter_by(date=target_date).all()
        return records
    
    def get_attendance_by_student(self, student_id, start_date=None, end_date=None):
        """Get attendance records for a specific student"""
        query = self.session.query(Attendance).filter_by(student_id=student_id)
        
        if start_date:
            query = query.filter(Attendance.date >= start_date)
        if end_date:
            query = query.filter(Attendance.date <= end_date)
        
        return query.all()

    def get_attendance_report(self, start_date=None, end_date=None):
        """Generate attendance report for a date range as a pandas DataFrame.

        Columns: Date, Student ID, Name, Time, Status, Confidence
        """
        query = self.session.query(Attendance)

        if start_date:
            query = query.filter(Attendance.date >= start_date)
        if end_date:
            query = query.filter(Attendance.date <= end_date)

        records = query.all()

        # Convert to DataFrame for easier consumption
        data = []
        for record in records:
            data.append({
                'Date': record.date,
                'Student ID': record.student_id,
                'Name': record.recognized_name or record.student_id,
                'Time': record.time,
                'Status': record.status,
                'Confidence': record.confidence
            })

        if data:
            return pd.DataFrame(data)
        else:
            # Return empty df with expected columns
            return pd.DataFrame(columns=['Date', 'Student ID', 'Name', 'Time', 'Status', 'Confidence'])
    
    def get_daily_report(self, target_date=None):
        """Get daily attendance report"""
        if not target_date:
            target_date = date.today()
        
        # Get all attendance for the day
        attendance_records = self.get_attendance_by_date(target_date)
        
        # Get all active students
        all_students = self.get_all_students()
        
        # Create report
        report = []
        present_ids = [record.student_id for record in attendance_records]
        
        for student in all_students:
            status = "Present" if student.student_id in present_ids else "Absent"
            
            # Get attendance record if present
            attendance_record = next(
                (record for record in attendance_records if record.student_id == student.student_id),
                None
            )
            
            report.append({
                'student_id': student.student_id,
                'name': student.name,
                'department': student.department or 'N/A',
                'status': status,
                'time': attendance_record.time if attendance_record else None,
                'confidence': attendance_record.confidence if attendance_record else None
            })
        
        return pd.DataFrame(report)
    
    def delete_student(self, student_id):
        """Delete a student (set inactive - SOFT DELETE)"""
        try:
            student = self.session.query(Student).filter_by(student_id=student_id).first()
            if student:
                student.is_active = False
                self.session.commit()
                print(f"✓ Student {student_id} marked as inactive.")
                return True
            else:
                print(f"✗ Student {student_id} not found.")
                return False
        except Exception as e:
            print(f"✗ Error deleting student: {e}")
            self.session.rollback()
            return False
    
    def permanent_delete_student(self, student_id):
        """Permanently delete a student and all related attendance records (HARD DELETE)"""
        try:
            # Delete attendance records first
            attendance_count = self.session.query(Attendance).filter_by(student_id=student_id).delete()
            
            # Delete student
            student = self.session.query(Student).filter_by(student_id=student_id).first()
            if student:
                self.session.delete(student)
                self.session.commit()
                print(f"✓ Student {student_id} permanently deleted.")
                print(f"✓ Removed {attendance_count} attendance records.")
                return True
            else:
                print(f"✗ Student {student_id} not found.")
                return False
        except Exception as e:
            print(f"✗ Error permanently deleting student: {e}")
            self.session.rollback()
            return False
    
    def close(self):
        """Close database session"""
        self.session.close()

# Create global instance
db_handler = DatabaseHandler()