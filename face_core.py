# face_core.py - Core face recognition functionality
import cv2
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, date
from pathlib import Path
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from abc import ABC, abstractmethod
import pickle
import json

class BaseClassifier(ABC):
    """Abstract base class for face classifiers"""
    
    @abstractmethod
    def train(self, embeddings, names):
        """Train the classifier with embeddings and corresponding names"""
        pass
    
    @abstractmethod
    def predict(self, embedding):
        """Predict identity from embedding. Returns (name, confidence)"""
        pass
    
    @abstractmethod
    def save(self, filepath):
        """Save trained classifier to file"""
        pass
    
    @abstractmethod
    def load(self, filepath):
        """Load trained classifier from file"""
        pass
    
    @abstractmethod
    def get_name(self):
        """Return classifier name for logging"""
        pass

class FaceEmbeddingExtractor:
    """Handles face detection and embedding extraction"""
    
    def __init__(self, device=None):
        # Device setup
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize face detection and recognition models
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709, 
            post_process=True,
            device=self.device
        )
        
        # FaceNet model for embeddings
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    
    def extract_face_embedding(self, image):
        """
        Extract face embedding from image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            embedding: 512-dimensional face embedding or None
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect and extract face
        face_tensor = self.mtcnn(rgb_image)
        
        if face_tensor is not None:
            # Generate embedding
            with torch.no_grad():
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                embedding = self.facenet(face_tensor).cpu().numpy().flatten()
            return embedding
        
        return None
    
    def detect_faces(self, image):
        """
        Detect faces in image and return bounding boxes
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            boxes: List of face bounding boxes or None
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(rgb_image)
        return boxes

class AttendanceDatabase:
    """Handles all database operations"""
    
    def __init__(self, database_path='attendance.db'):
        self.database_path = database_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for attendance records"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                date DATE NOT NULL,
                confidence REAL NOT NULL,
                classifier_type TEXT DEFAULT 'unknown'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enrolled_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                enrollment_date DATETIME NOT NULL,
                total_embeddings INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_enrolled_user(self, name, num_embeddings):
        """Add or update enrolled user"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO enrolled_users (name, enrollment_date, total_embeddings)
            VALUES (?, ?, ?)
        ''', (name, datetime.now(), num_embeddings))
        
        conn.commit()
        conn.close()
    
    def is_already_marked_today(self, name):
        """Check if person already marked attendance today"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        today = date.today()
        cursor.execute('''
            SELECT COUNT(*) FROM attendance 
            WHERE name = ? AND date = ?
        ''', (name, today))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def mark_attendance(self, name, confidence, classifier_type="unknown"):
        """Mark attendance for a person"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        cursor.execute('''
            INSERT INTO attendance (name, timestamp, date, confidence, classifier_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, now, now.date(), confidence, classifier_type))
        
        conn.commit()
        conn.close()
        
        return now
    
    def get_attendance_report(self, start_date=None, end_date=None):
        """Generate attendance report"""
        conn = sqlite3.connect(self.database_path)
        
        query = "SELECT * FROM attendance"
        params = []
        
        if start_date and end_date:
            query += " WHERE date BETWEEN ? AND ?"
            params = [start_date, end_date]
        elif start_date:
            query += " WHERE date >= ?"
            params = [start_date]
        elif end_date:
            query += " WHERE date <= ?"
            params = [end_date]
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_enrolled_users(self):
        """Get list of enrolled users"""
        conn = sqlite3.connect(self.database_path)
        df = pd.read_sql_query("SELECT * FROM enrolled_users", conn)
        conn.close()
        return df

class FaceEnrollmentSystem:
    """Handles enrollment of new people"""
    
    def __init__(self, extractor, database):
        self.extractor = extractor
        self.database = database
    
    def enroll_from_webcam(self, name, num_samples=10):
        """Enroll person using webcam capture"""
        embeddings = []
        
        print(f"Capturing {num_samples} samples for {name} from webcam...")
        print("Press SPACE to capture, ESC to cancel, ENTER when done")
        
        cap = cv2.VideoCapture(0)
        captured_count = 0
        
        while captured_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, 
                      f"Enrolling: {name} ({captured_count}/{num_samples})", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, 
                      "Press SPACE to capture", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw face detection box
            boxes = self.extractor.detect_faces(frame)
            
            if boxes is not None:
                for box in boxes:
                    cv2.rectangle(display_frame, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                (0, 255, 0), 2)
            
            cv2.imshow('Enrollment', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                embedding = self.extractor.extract_face_embedding(frame)
                if embedding is not None:
                    embeddings.append(embedding)
                    captured_count += 1
                    print(f"Captured sample {captured_count}/{num_samples}")
                else:
                    print("No face detected, try again")
            elif key == 13:  # ENTER
                if captured_count >= 3:  # Minimum 3 samples
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(embeddings) == 0:
            print(f"No valid face embeddings found for {name}")
            return []
        
        # Update database
        self.database.add_enrolled_user(name, len(embeddings))
        
        print(f"Successfully enrolled {name} with {len(embeddings)} samples")
        return embeddings
    
    def enroll_from_images(self, name, images_path):
        """Enroll person from image folder"""
        embeddings = []
        
        print(f"Processing images from {images_path}...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for image_file in Path(images_path).iterdir():
            if image_file.suffix.lower() in image_extensions:
                image = cv2.imread(str(image_file))
                if image is not None:
                    embedding = self.extractor.extract_face_embedding(image)
                    if embedding is not None:
                        embeddings.append(embedding)
                        print(f"Processed: {image_file.name}")
        
        if len(embeddings) == 0:
            print(f"No valid face embeddings found for {name}")
            return []
        
        # Update database
        self.database.add_enrolled_user(name, len(embeddings))
        
        print(f"Successfully enrolled {name} with {len(embeddings)} samples")
        return embeddings

class FaceRecognitionSystem:
    """Main face recognition system that ties everything together"""
    
    def __init__(self, classifier, model_path='models/', database_path='attendance.db', 
                 confidence_threshold=0.7, device=None):
        """
        Initialize Face Recognition System
        
        Args:
            classifier: Instance of BaseClassifier
            model_path: Path to save/load models
            database_path: Path to SQLite database
            confidence_threshold: Minimum confidence for recognition
            device: PyTorch device
        """
        self.classifier = classifier
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.extractor = FaceEmbeddingExtractor(device)
        self.database = AttendanceDatabase(database_path)
        self.enrollment_system = FaceEnrollmentSystem(self.extractor, self.database)
        
        # Storage for embeddings and names
        self.known_embeddings = []
        self.known_names = []
        
        # Load existing model
        self.load_system()
    
    def enroll_person(self, name, images_path=None, webcam_capture=False, num_samples=10):
        """Enroll a new person"""
        if webcam_capture:
            new_embeddings = self.enrollment_system.enroll_from_webcam(name, num_samples)
        elif images_path:
            new_embeddings = self.enrollment_system.enroll_from_images(name, images_path)
        else:
            print("Please specify either images_path or set webcam_capture=True")
            return False
        
        if len(new_embeddings) == 0:
            return False
        
        # Add to known embeddings
        self.known_embeddings.extend(new_embeddings)
        self.known_names.extend([name] * len(new_embeddings))
        
        # Train classifier
        success = self.train_classifier()
        if success:
            self.save_system()
        
        return success
    
    def train_classifier(self):
        """Train the classifier with current embeddings"""
        if len(self.known_embeddings) == 0:
            print("No embeddings to train classifier")
            return False
        
        print(f"Training {self.classifier.get_name()} with {len(self.known_embeddings)} samples...")
        success = self.classifier.train(self.known_embeddings, self.known_names)
        
        if success:
            unique_people = len(set(self.known_names))
            print(f"Successfully trained {self.classifier.get_name()} with {unique_people} people")
        
        return success
    
    def predict_person(self, embedding):
        """Predict person from embedding"""
        name, confidence = self.classifier.predict(embedding)
        
        if confidence >= self.confidence_threshold:
            return name, confidence
        else:
            return "Unknown", confidence
    
    def mark_attendance(self, name, confidence):
        """Mark attendance for a person"""
        if self.database.is_already_marked_today(name):
            return f"Already marked today, {name}!", False
        
        timestamp = self.database.mark_attendance(name, confidence, self.classifier.get_name())
        print(f"Attendance marked for {name} at {timestamp.strftime('%H:%M:%S')} using {self.classifier.get_name()}")
        return f"Welcome, {name}!", True
    
    def save_system(self):
        """Save the complete system state"""
        # Save classifier
        classifier_path = self.model_path / f"{self.classifier.get_name()}_model.pkl"
        self.classifier.save(classifier_path)
        
        # Save embeddings and names
        system_data = {
            'known_embeddings': self.known_embeddings,
            'known_names': self.known_names,
            'confidence_threshold': self.confidence_threshold,
            'classifier_type': self.classifier.get_name()
        }
        
        system_path = self.model_path / 'system_data.pkl'
        with open(system_path, 'wb') as f:
            pickle.dump(system_data, f)
        
        print(f"System saved successfully using {self.classifier.get_name()}")
    
    def load_system(self):
        """Load the complete system state"""
        system_path = self.model_path / 'system_data.pkl'
        
        if system_path.exists():
            try:
                with open(system_path, 'rb') as f:
                    system_data = pickle.load(f)
                
                self.known_embeddings = system_data['known_embeddings']
                self.known_names = system_data['known_names']
                
                # Load classifier - try current classifier first
                classifier_path = self.model_path / f"{self.classifier.get_name()}_model.pkl"
                if classifier_path.exists():
                    self.classifier.load(classifier_path)
                    unique_people = len(set(self.known_names))
                    print(f"System loaded successfully with {unique_people} enrolled users using {self.classifier.get_name()}")
                    return True
                else:
                    print(f"Classifier model file not found: {classifier_path}")
                    print("You may need to retrain the classifier or switch to a different one")
                    
            except Exception as e:
                print(f"Error loading system: {e}")
                return False
        
        print("No existing system found")
        return False
    
    def run_live_attendance(self):
        """Run live attendance system with webcam"""
        print(f"Starting Face Recognition Attendance System using {self.classifier.get_name()}...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        cap = cv2.VideoCapture(0)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame to improve performance
            frame_count += 1
            if frame_count % 5 != 0:
                cv2.imshow('Attendance System', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Create display frame
            display_frame = frame.copy()
            
            # Detect faces and draw boxes
            boxes = self.extractor.detect_faces(frame)
            
            if boxes is not None:
                for box in boxes:
                    # Draw face box
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Extract face for recognition
                    face_crop = frame[max(0, y1):min(frame.shape[0], y2), 
                                    max(0, x1):min(frame.shape[1], x2)]
                    
                    if face_crop.size > 0:
                        # Get embedding and predict
                        embedding = self.extractor.extract_face_embedding(face_crop)
                        
                        if embedding is not None:
                            predicted_name, confidence = self.predict_person(embedding)
                            
                            # Choose color based on recognition
                            if predicted_name != "Unknown":
                                color = (0, 255, 0)  # Green for recognized
                                
                                # Try to mark attendance
                                if confidence >= self.confidence_threshold:
                                    message, marked = self.mark_attendance(predicted_name, confidence)
                                else:
                                    message = f"Low confidence: {predicted_name}"
                            else:
                                color = (0, 0, 255)  # Red for unknown
                                message = "Unknown Person"
                            
                            # Draw rectangle and text
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw name and confidence
                            label = f"{predicted_name} ({confidence:.2f})"
                            cv2.putText(display_frame, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        else:
                            # Face detected but no embedding extracted
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(display_frame, "Processing...", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add system info
            unique_people = len(set(self.known_names)) if self.known_names else 0
            cv2.putText(display_frame, f"Enrolled: {unique_people} | Classifier: {self.classifier.get_name()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Threshold: {self.confidence_threshold:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Attendance System', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshot_{timestamp}.jpg", display_frame)
                print(f"Screenshot saved: screenshot_{timestamp}.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def generate_report(self, start_date=None, end_date=None):
        """Generate attendance report"""
        return self.database.get_attendance_report(start_date, end_date)
    
    def switch_classifier(self, new_classifier):
        """
        Switch to a different classifier while keeping the same enrolled data
        
        Args:
            new_classifier: Instance of BaseClassifier
            
        Returns:
            bool: Success status
        """
        if not self.known_embeddings:
            print("No enrolled data found. Please enroll people first.")
            return False
        
        print(f"Switching from {self.classifier.get_name()} to {new_classifier.get_name()}")
        
        # Try to load existing model for the new classifier
        old_classifier = self.classifier
        self.classifier = new_classifier
        
        classifier_path = self.model_path / f"{new_classifier.get_name()}_model.pkl"
        
        if classifier_path.exists():
            # Load existing trained model
            success = self.classifier.load(classifier_path)
            if success:
                print(f"Loaded existing {new_classifier.get_name()} model")
                return True
        
        # Train new classifier with existing data
        print(f"Training {new_classifier.get_name()} with existing enrolled data...")
        success = self.train_classifier()
        
        if success:
            self.save_system()
            print(f"Successfully switched to {new_classifier.get_name()}")
            return True
        else:
            # Revert to old classifier if training failed
            self.classifier = old_classifier
            print(f"Failed to switch classifiers, reverted to {old_classifier.get_name()}")
            return False


