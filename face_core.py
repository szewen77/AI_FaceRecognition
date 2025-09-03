# face_core.py - Streamlined Face Recognition with Multi-Classifier System
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
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional

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
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Validate image
            if rgb_image is None or rgb_image.size == 0:
                return None
            
            # Detect and extract face
            face_tensor = self.mtcnn(rgb_image)
            
            if face_tensor is not None:
                # Generate embedding
                with torch.no_grad():
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)
                    embedding = self.facenet(face_tensor).cpu().numpy().flatten()
                return embedding
            
            return None
        except Exception as e:
            print(f"Error in extract_face_embedding: {str(e)}")
            return None
    
    def detect_faces(self, image):
        """
        Detect faces in image and return bounding boxes
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            boxes: List of face bounding boxes or None
        """
        try:
            # Validate image
            if image is None or image.size == 0:
                return None
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Validate converted image
            if rgb_image is None or rgb_image.size == 0:
                return None
                
            boxes, _ = self.mtcnn.detect(rgb_image)
            return boxes
        except Exception as e:
            print(f"Error in detect_faces: {str(e)}")
            return None

class FaceVerifier:
    """Handles face verification for enrollment validation"""
    
    def __init__(self, similarity_threshold=0.7):
        """
        Initialize face verifier
        
        Args:
            similarity_threshold: Minimum cosine similarity for same person verification
        """
        self.similarity_threshold = similarity_threshold
    
    def verify_same_person(self, new_embeddings: List[np.ndarray], 
                          existing_embeddings: List[np.ndarray]) -> Tuple[bool, float]:
        """
        Verify if new embeddings belong to the same person as existing embeddings
        
        Args:
            new_embeddings: List of new face embeddings
            existing_embeddings: List of existing face embeddings for the person
            
        Returns:
            (is_same_person, average_similarity)
        """
        if not new_embeddings or not existing_embeddings:
            return False, 0.0
        
        similarities = []
        
        # Compare each new embedding with all existing embeddings
        for new_emb in new_embeddings:
            max_similarity = 0.0
            for existing_emb in existing_embeddings:
                # Reshape for cosine similarity calculation
                new_emb_reshaped = new_emb.reshape(1, -1)
                existing_emb_reshaped = existing_emb.reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(new_emb_reshaped, existing_emb_reshaped)[0][0]
                max_similarity = max(max_similarity, similarity)
            
            similarities.append(max_similarity)
        
        average_similarity = np.mean(similarities)
        is_same_person = average_similarity >= self.similarity_threshold
        
        return is_same_person, average_similarity

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
                classifier_type TEXT DEFAULT 'multi_classifier'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enrolled_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                enrollment_date DATETIME NOT NULL,
                total_embeddings INTEGER DEFAULT 0,
                last_updated DATETIME NOT NULL
            )
        ''')
        
        # Add embeddings storage table for verification
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_date DATETIME NOT NULL,
                FOREIGN KEY (name) REFERENCES enrolled_users (name)
            )
        ''')
        
        # Check if last_updated column exists, if not add it (for existing databases)
        try:
            cursor.execute("SELECT last_updated FROM enrolled_users LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            print("Adding last_updated column to existing database...")
            cursor.execute("ALTER TABLE enrolled_users ADD COLUMN last_updated DATETIME")
            # Update existing records with current timestamp
            cursor.execute("UPDATE enrolled_users SET last_updated = enrollment_date WHERE last_updated IS NULL")
        
        conn.commit()
        conn.close()
    
    def add_enrolled_user(self, name, embeddings):
        """Add or update enrolled user with embeddings"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT total_embeddings FROM enrolled_users WHERE name = ?', (name,))
        result = cursor.fetchone()
        
        if result:
            # Update existing user
            old_count = result[0]
            new_count = old_count + len(embeddings)
            cursor.execute('''
                UPDATE enrolled_users 
                SET total_embeddings = ?, last_updated = ?
                WHERE name = ?
            ''', (new_count, datetime.now(), name))
        else:
            # Insert new user
            cursor.execute('''
                INSERT INTO enrolled_users (name, enrollment_date, total_embeddings, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (name, datetime.now(), len(embeddings), datetime.now()))
        
        # Add embeddings to storage
        for embedding in embeddings:
            cursor.execute('''
                INSERT INTO user_embeddings (name, embedding, created_date)
                VALUES (?, ?, ?)
            ''', (name, pickle.dumps(embedding), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_user_embeddings(self, name):
        """Get all embeddings for a specific user"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT embedding FROM user_embeddings WHERE name = ?
        ''', (name,))
        
        results = cursor.fetchall()
        conn.close()
        
        embeddings = []
        for result in results:
            embedding = pickle.loads(result[0])
            embeddings.append(embedding)
        
        return embeddings
    
    def user_exists(self, name):
        """Check if user already exists in database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM enrolled_users WHERE name = ?', (name,))
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
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
    
    def mark_attendance(self, name, confidence, classifier_type="multi_classifier"):
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
        df = pd.read_sql_query("SELECT * FROM enrolled_users ORDER BY last_updated DESC", conn)
        conn.close()
        return df
    
    def delete_enrolled_user(self, name):
        """Delete an enrolled user and all their data"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Check if user exists
            cursor.execute('SELECT COUNT(*) FROM enrolled_users WHERE name = ?', (name,))
            if cursor.fetchone()[0] == 0:
                conn.close()
                return False, "User not found"
            
            # Delete user embeddings
            cursor.execute('DELETE FROM user_embeddings WHERE name = ?', (name,))
            embeddings_deleted = cursor.rowcount
            
            # Delete attendance records
            cursor.execute('DELETE FROM attendance WHERE name = ?', (name,))
            attendance_deleted = cursor.rowcount
            
            # Delete user from enrolled_users
            cursor.execute('DELETE FROM enrolled_users WHERE name = ?', (name,))
            user_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            return True, f"Deleted user '{name}': {embeddings_deleted} embeddings, {attendance_deleted} attendance records"
            
        except Exception as e:
            conn.rollback()
            conn.close()
            return False, f"Error deleting user: {str(e)}"
    
    def get_all_enrolled_user_names(self):
        """Get list of all enrolled user names"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT name FROM enrolled_users ORDER BY name')
        names = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return names

class FaceEnrollmentSystem:
    """Handles enrollment of new people with verification"""
    
    def __init__(self, extractor, database, verifier):
        self.extractor = extractor
        self.database = database
        self.verifier = verifier
    
    def enroll_from_webcam(self, name, num_samples=10):
        """Enroll person using webcam capture with verification for existing users"""
        embeddings = []
        
        # Check if user already exists
        user_exists = self.database.user_exists(name)
        if user_exists:
            print(f"User '{name}' already exists. Adding new samples for enhancement...")
            existing_embeddings = self.database.get_user_embeddings(name)
        else:
            print(f"Enrolling new user: {name}")
            existing_embeddings = []
        
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
            status = "Adding samples" if user_exists else "New enrollment"
            cv2.putText(display_frame, 
                      f"{status}: {name} ({captured_count}/{num_samples})", 
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
        
        # Verify if adding to existing user
        if user_exists and existing_embeddings:
            is_same_person, similarity = self.verifier.verify_same_person(embeddings, existing_embeddings)
            
            if not is_same_person:
                print(f"ERROR: New images do not match existing user '{name}'!")
                print(f"Similarity score: {similarity:.3f} (threshold: {self.verifier.similarity_threshold})")
                print("Enrollment rejected. Original data retained.")
                return []
            else:
                print(f"✓ Verification passed! Similarity: {similarity:.3f}")
                print(f"Adding {len(embeddings)} new samples to existing user '{name}'")
        
        # Update database
        self.database.add_enrolled_user(name, embeddings)
        
        total_samples = len(existing_embeddings) + len(embeddings) if user_exists else len(embeddings)
        print(f"Successfully enrolled {name} with {total_samples} total samples")
        return embeddings
    
    def enroll_from_images(self, name, images_path):
        """Enroll person from image folder with verification for existing users"""
        embeddings = []
        
        # Check if user already exists
        user_exists = self.database.user_exists(name)
        if user_exists:
            print(f"User '{name}' already exists. Adding new samples for enhancement...")
            existing_embeddings = self.database.get_user_embeddings(name)
        else:
            print(f"Enrolling new user: {name}")
            existing_embeddings = []
        
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
        
        # Verify if adding to existing user
        if user_exists and existing_embeddings:
            is_same_person, similarity = self.verifier.verify_same_person(embeddings, existing_embeddings)
            
            if not is_same_person:
                print(f"ERROR: New images do not match existing user '{name}'!")
                print(f"Similarity score: {similarity:.3f} (threshold: {self.verifier.similarity_threshold})")
                print("Enrollment rejected. Original data retained.")
                return []
            else:
                print(f"✓ Verification passed! Similarity: {similarity:.3f}")
                print(f"Adding {len(embeddings)} new samples to existing user '{name}'")
        
        # Update database
        self.database.add_enrolled_user(name, embeddings)
        
        total_samples = len(existing_embeddings) + len(embeddings) if user_exists else len(embeddings)
        print(f"Successfully enrolled {name} with {total_samples} total samples")
        return embeddings
    
    def enroll_from_file_list(self, name, file_paths):
        """Enroll person from a list of image files with verification for existing users"""
        embeddings = []
        
        # Check if user already exists
        user_exists = self.database.user_exists(name)
        if user_exists:
            print(f"User '{name}' already exists. Adding new samples for enhancement...")
            existing_embeddings = self.database.get_user_embeddings(name)
        else:
            print(f"Enrolling new user: {name}")
            existing_embeddings = []
        
        print(f"Processing {len(file_paths)} selected image file(s)...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for file_path in file_paths:
            file_path = Path(file_path)
            if file_path.suffix.lower() in image_extensions:
                try:
                    image = cv2.imread(str(file_path))
                    if image is not None:
                        embedding = self.extractor.extract_face_embedding(image)
                        if embedding is not None:
                            embeddings.append(embedding)
                            print(f"Processed: {file_path.name}")
                        else:
                            print(f"No face found in: {file_path.name}")
                    else:
                        print(f"Could not load image: {file_path.name}")
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
            else:
                print(f"Skipped unsupported file: {file_path.name}")
        
        if len(embeddings) == 0:
            print(f"No valid face embeddings found for {name}")
            return []
        
        # Verify if adding to existing user
        if user_exists and existing_embeddings:
            is_same_person, similarity = self.verifier.verify_same_person(embeddings, existing_embeddings)
            
            if not is_same_person:
                print(f"ERROR: New images do not match existing user '{name}'!")
                print(f"Similarity score: {similarity:.3f} (threshold: {self.verifier.similarity_threshold})")
                print("Enrollment rejected. Original data retained.")
                return []
            else:
                print(f"✓ Verification passed! Similarity: {similarity:.3f}")
                print(f"Adding {len(embeddings)} new samples to existing user '{name}'")
        
        # Update database
        self.database.add_enrolled_user(name, embeddings)
        
        total_samples = len(existing_embeddings) + len(embeddings) if user_exists else len(embeddings)
        print(f"Successfully enrolled {name} with {total_samples} total samples")
        return embeddings

class FixedMultiClassifierSystem:
    """Multi-Classifier System with simultaneous training and majority voting"""
    
    def __init__(self, model_path='models/', database_path='attendance.db', 
                 confidence_threshold=0.7, verification_threshold=0.7, device=None):
        """
        Initialize Fixed Multi-Classifier System
        
        Args:
            model_path: Path to save/load models
            database_path: Path to SQLite database
            confidence_threshold: Minimum confidence for recognition
            verification_threshold: Minimum similarity for face verification
            device: PyTorch device
        """
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.extractor = FaceEmbeddingExtractor(device)
        self.database = AttendanceDatabase(database_path)
        self.verifier = FaceVerifier(verification_threshold)
        self.enrollment_system = FaceEnrollmentSystem(self.extractor, self.database, self.verifier)
        
        # Initialize all three classifiers
        print("🔄 Loading classifiers...")
        try:
            from classifiers.svm_classifier import SVMClassifier
            from classifiers.knn_classifier import KNNClassifier
            from classifiers.logistic_regression import LogisticRegressionClassifier
            
            self.classifiers = {
                'SVM': SVMClassifier(),
                'KNN': KNNClassifier(), 
                'LogisticRegression': LogisticRegressionClassifier()
            }
            print("✅ All classifiers loaded successfully")
            
        except ImportError as e:
            print(f"❌ Error loading classifiers: {e}")
            print("Please ensure classifier files are in the 'classifiers/' directory")
            raise
        
        # Performance tracking for each classifier
        self.classifier_performance = {
            'SVM': {'correct': 0, 'total': 0, 'predictions': []},
            'KNN': {'correct': 0, 'total': 0, 'predictions': []},
            'LogisticRegression': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        # Storage for embeddings and names
        self.known_embeddings = []
        self.known_names = []
        
        # Load existing system
        print("🔄 Loading existing system data...")
        self.load_system()
    
    def enroll_person(self, name, images_path=None, webcam_capture=False, num_samples=10, file_list=None):
        """Enroll person with simultaneous multi-classifier training"""
        print(f"\n=== ENROLLING: {name} ===")
        
        # Get new embeddings using enrollment system (with verification)
        if webcam_capture:
            new_embeddings = self.enrollment_system.enroll_from_webcam(name, num_samples)
        elif images_path:
            new_embeddings = self.enrollment_system.enroll_from_images(name, images_path)
        elif file_list:
            new_embeddings = self.enrollment_system.enroll_from_file_list(name, file_list)
        else:
            print("Please specify either images_path, file_list or set webcam_capture=True")
            return False
        
        if len(new_embeddings) == 0:
            return False
        
        print(f"✅ Got {len(new_embeddings)} new embeddings for {name}")
        
        # Rebuild all embeddings from database
        self._rebuild_embeddings_from_database()
        
        print(f"📊 Total dataset now: {len(self.known_embeddings)} embeddings, {len(set(self.known_names))} people")
        
        # Update all classifiers with new data
        self._update_all_classifiers_with_new_data()
        
        # Save updated system
        self.save_system()
        
        return True
    
    def _update_all_classifiers_with_new_data(self):
        """Update each classifier according to the new workflow"""
        if len(self.known_embeddings) == 0:
            print("No embeddings to train with")
            return False
        
        print("\n🔄 UPDATING ALL CLASSIFIERS...")
        
        success_count = 0
        
        # Update each classifier
        for clf_name, classifier in self.classifiers.items():
            try:
                print(f"\nUpdating {clf_name}...")
                
                if clf_name == 'KNN':
                    # KNN: Just add embeddings to dataset (no retraining needed)
                    print("  → Adding embeddings to KNN dataset")
                    success = classifier.train(self.known_embeddings, self.known_names)
                    
                elif clf_name in ['SVM', 'LogisticRegression']:
                    # SVM & LogReg: Full retraining with old + new data
                    print(f"  → Retraining {clf_name} with all data")
                    success = classifier.train(self.known_embeddings, self.known_names)
                
                if success:
                    print(f"  ✅ {clf_name} updated successfully")
                    success_count += 1
                else:
                    print(f"  ❌ {clf_name} update failed")
                    
            except Exception as e:
                print(f"  ❌ {clf_name} error: {e}")
        
        print(f"\n📈 TRAINING RESULTS: {success_count}/3 classifiers updated successfully")
        
        # Ensure all classifiers trained on same data for fair comparison
        unique_people = len(set(self.known_names))
        print(f"✅ All classifiers now trained on {len(self.known_embeddings)} samples from {unique_people} people")
        
        return success_count > 0
    
    def predict_with_voting(self, embedding, debug=False):
        """
        Predict using all classifiers and majority voting
        Records each classifier's prediction separately
        
        Returns:
            (final_prediction, confidence, individual_results, voting_details)
        """
        individual_results = {}
        voting_details = {'votes': {}, 'confidences': {}}
        
        if debug:
            print(f"\n=== MULTI-CLASSIFIER PREDICTION ===")
        
        # Get prediction from each classifier
        for clf_name, classifier in self.classifiers.items():
            try:
                name, confidence = classifier.predict(embedding)
                individual_results[clf_name] = {
                    'name': name,
                    'confidence': confidence
                }
                
                # Record for voting (skip Unknown predictions)
                if name != "Unknown":
                    if name not in voting_details['votes']:
                        voting_details['votes'][name] = 0
                        voting_details['confidences'][name] = []
                    
                    voting_details['votes'][name] += 1
                    voting_details['confidences'][name].append(confidence)
                
                if debug:
                    print(f"  {clf_name}: {name} ({confidence:.3f})")
                    
            except Exception as e:
                print(f"  ❌ {clf_name} prediction error: {e}")
                individual_results[clf_name] = {
                    'name': 'Unknown',
                    'confidence': 0.0
                }
        
        # Majority voting
        if voting_details['votes']:
            # Find name with most votes
            winner_name = max(voting_details['votes'].keys(), key=lambda k: voting_details['votes'][k])
            winner_votes = voting_details['votes'][winner_name]
            
            # Calculate average confidence for the winning name
            winner_confidences = voting_details['confidences'][winner_name]
            avg_confidence = sum(winner_confidences) / len(winner_confidences) if winner_confidences else 0.0
            
            if debug:
                print(f"\n📊 VOTING RESULTS:")
                for name, votes in voting_details['votes'].items():
                    avg_conf = sum(voting_details['confidences'][name]) / len(voting_details['confidences'][name])
                    print(f"  {name}: {votes} votes, avg confidence: {avg_conf:.3f}")
                print(f"🏆 WINNER: {winner_name} ({winner_votes} votes, {avg_confidence:.3f} confidence)")
        else:
            winner_name = 'Unknown'
            avg_confidence = 0.0
        
        return winner_name, avg_confidence, individual_results, voting_details
    
    def mark_attendance_with_tracking(self, name, confidence, individual_results, voting_details):
        """Mark attendance and track each classifier's performance"""
        if self.database.is_already_marked_today(name):
            return f"Already marked today, {name}!", False
        
        # Record individual predictions for performance tracking
        for clf_name, result in individual_results.items():
            prediction_record = {
                'timestamp': datetime.now(),
                'predicted_name': result['name'],
                'actual_name': name,  # Assuming the voted result is correct
                'confidence': result['confidence'],
                'correct': result['name'] == name
            }
            
            self.classifier_performance[clf_name]['predictions'].append(prediction_record)
            self.classifier_performance[clf_name]['total'] += 1
            
            if result['name'] == name:
                self.classifier_performance[clf_name]['correct'] += 1
        
        # Create detailed classifier info for database
        classifier_details = []
        for clf_name, result in individual_results.items():
            classifier_details.append(f"{clf_name}:{result['name']}({result['confidence']:.3f})")
        
        voting_info = f"Votes:{voting_details['votes']}"
        classifier_info = f"Individual:[{','.join(classifier_details)}] | {voting_info}"
        
        # Mark attendance in database
        timestamp = self.database.mark_attendance(name, confidence, classifier_info)
        
        print(f"✅ Attendance marked for {name} at {timestamp.strftime('%H:%M:%S')}")
        print(f"   Final decision: {name} ({confidence:.3f}) via majority voting")
        
        return f"Welcome, {name}!", True
    
    def get_classifier_performance_report(self):
        """Generate performance report for each classifier"""
        report = "=== CLASSIFIER PERFORMANCE TRACKING ===\n\n"
        
        for clf_name, perf in self.classifier_performance.items():
            if perf['total'] > 0:
                accuracy = (perf['correct'] / perf['total']) * 100
                report += f"{clf_name}:\n"
                report += f"  Predictions: {perf['total']}\n"
                report += f"  Correct: {perf['correct']}\n"
                report += f"  Accuracy: {accuracy:.2f}%\n"
                
                # Recent predictions
                recent = perf['predictions'][-5:] if len(perf['predictions']) > 5 else perf['predictions']
                if recent:
                    report += f"  Recent predictions:\n"
                    for pred in recent:
                        status = "✅" if pred['correct'] else "❌"
                        report += f"    {status} {pred['predicted_name']} ({pred['confidence']:.3f})\n"
                report += "\n"
            else:
                report += f"{clf_name}: No predictions yet\n\n"
        
        return report
    
    def _rebuild_embeddings_from_database(self):
        """Rebuild embeddings and names from database"""
        self.known_embeddings = []
        self.known_names = []
        
        try:
            enrolled_users = self.database.get_enrolled_users()
            
            for _, user in enrolled_users.iterrows():
                name = user['name']
                embeddings = self.database.get_user_embeddings(name)
                
                self.known_embeddings.extend(embeddings)
                self.known_names.extend([name] * len(embeddings))
                
        except Exception as e:
            print(f"Error rebuilding embeddings: {e}")
    
    def delete_enrolled_user(self, name):
        """Delete user and update all classifiers"""
        success, message = self.database.delete_enrolled_user(name)
        
        if success:
            print(f"🗑️ Deleted {name} from database")
            
            # Rebuild embeddings
            self._rebuild_embeddings_from_database()
            
            # Update all classifiers with remaining data
            if len(self.known_embeddings) > 0:
                self._update_all_classifiers_with_new_data()
                self.save_system()
                return True, f"{message}. All classifiers updated successfully."
            else:
                # No users left
                self.known_embeddings = []
                self.known_names = []
                return True, f"{message}. No users remaining - system cleared."
        
        return False, message
    
    def save_system(self):
        """Save all classifiers and system state"""
        # Save each classifier
        save_results = {}
        for clf_name, classifier in self.classifiers.items():
            try:
                classifier_path = self.model_path / f"{clf_name}_model.pkl"
                classifier.save(classifier_path)
                save_results[clf_name] = True
                print(f"💾 Saved {clf_name} model")
            except Exception as e:
                print(f"❌ Error saving {clf_name}: {e}")
                save_results[clf_name] = False
        
        # Save system data
        system_data = {
            'known_embeddings': self.known_embeddings,
            'known_names': self.known_names,
            'confidence_threshold': self.confidence_threshold,
            'verification_threshold': self.verifier.similarity_threshold,
            'classifier_performance': self.classifier_performance
        }
        
        system_path = self.model_path / 'multi_classifier_system_data.pkl'
        try:
            with open(system_path, 'wb') as f:
                pickle.dump(system_data, f)
            print(f"💾 System data saved")
        except Exception as e:
            print(f"❌ Error saving system data: {e}")
        
        success_count = sum(save_results.values())
        print(f"💾 Save complete: {success_count}/3 classifiers saved")
    
    def load_system(self):
        """Load all classifiers and system state"""
        system_path = self.model_path / 'multi_classifier_system_data.pkl'

        print("🔄 Rebuilding training data from database...")
        self._rebuild_embeddings_from_database()
        
        if system_path.exists():
            try:
                # Load system data
                with open(system_path, 'rb') as f:
                    system_data = pickle.load(f)
                
                # self.known_embeddings = system_data.get('known_embeddings', [])
                # self.known_names = system_data.get('known_names', [])
                self.classifier_performance = system_data.get('classifier_performance', {
                    'SVM': {'correct': 0, 'total': 0, 'predictions': []},
                    'KNN': {'correct': 0, 'total': 0, 'predictions': []},
                    'LogisticRegression': {'correct': 0, 'total': 0, 'predictions': []}
                })
                
                # # Rebuild embeddings from database (more reliable than saved data)
                # self._rebuild_embeddings_from_database()
                
                # Load each classifier
                load_results = {}
                for clf_name, classifier in self.classifiers.items():
                    classifier_path = self.model_path / f"{clf_name}_model.pkl"
                    if classifier_path.exists():
                        try:
                            classifier.load(classifier_path)
                            load_results[clf_name] = True
                            print(f"✅ Loaded {clf_name} model")
                        except Exception as e:
                            print(f"❌ Error loading {clf_name}: {e}")
                            load_results[clf_name] = False
                    else:
                        print(f"⚠️  {clf_name} model file not found")
                        load_results[clf_name] = False
                
                success_count = sum(load_results.values())
                if success_count > 0:
                    unique_people = len(set(self.known_names)) if self.known_names else 0
                    print(f"🎯 Multi-Classifier System loaded: {success_count}/3 classifiers")
                    print(f"📊 Dataset: {len(self.known_embeddings)} embeddings, {unique_people} people")
                    return True
                    
            except Exception as e:
                print(f"❌ Error loading system: {e}")
        
        print("ℹ️  No existing system found - starting fresh")
        return False
    
    def run_live_attendance(self):
        """Run live attendance with multi-classifier voting"""
        print(f"🎥 Starting Multi-Classifier Live Attendance...")
        print(f"📊 Active classifiers: {list(self.classifiers.keys())}")
        print(f"🗳️  Using majority voting for final decisions")
        print("Controls: 'q'=quit, 's'=screenshot, 'p'=performance report, 'd'=debug mode")
    
        cap = cv2.VideoCapture(0)
        frame_count = 0
        detected_count = 0
        recognized_count = 0
        debug_mode = False
    
        # Check if system is ready
        if len(self.known_embeddings) == 0:
            print("⚠️  No enrolled users found! Please enroll users first.")
            cap.release()
            return
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            # Process every 10th frame for performance
            frame_count += 1
            if frame_count % 10 != 0:
                cv2.imshow('Multi-Classifier Attendance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
        
            # Create display frame
            display_frame = frame.copy()
        
            # Detect faces
            boxes = self.extractor.detect_faces(frame)
        
            if boxes is not None:
                detected_count += len(boxes)
            
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                
                    # Extract face for recognition
                    # Validate bounding box coordinates
                    if x1 >= x2 or y1 >= y2:
                        continue  # Skip invalid bounding box
                        
                    face_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                
                    # Validate face crop dimensions
                    if face_crop.size > 0 and face_crop.shape[0] > 20 and face_crop.shape[1] > 20:
                        # Get embedding
                        embedding = self.extractor.extract_face_embedding(face_crop)
                    
                        if embedding is not None:
                            # Multi-classifier prediction with voting
                            predicted_name, confidence, individual_results, voting_details = self.predict_with_voting(embedding, debug=debug_mode)
                        
                            attendance_marked = False
                        
                            if predicted_name != "Unknown" and confidence >= self.confidence_threshold:
                                # Only show name if both recognized AND confident enough
                                color = (0, 255, 0)  # Green - confident recognition
                                recognized_count += 1
                                message, attendance_marked = self.mark_attendance_with_tracking(predicted_name, confidence, individual_results, voting_details)
                                display_name = predicted_name
                                display_confidence = confidence
                            else:
                                # Low confidence or no recognition = treat as Unknown
                                color = (0, 0, 255)  # Red - unknown
                                display_name = "Unknown"
                                display_confidence = 0.0
                        
                            # Draw rectangle and label
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{display_name} ({display_confidence:.2f})"
                            cv2.putText(display_frame, label, (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                            # Show voting details only for recognized faces
                            if predicted_name != "Unknown" and confidence >= self.confidence_threshold:
                                vote_text = f"Votes: {voting_details['votes'].get(predicted_name, 0)}/3"
                                cv2.putText(display_frame, vote_text, (x1, y2+20),cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            
                                if attendance_marked:
                                    status_text = "✓ MARKED"
                                    cv2.putText(display_frame, status_text, (x1, y2+40),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                        
                            # Debug mode shows actual predictions even when displayed as Unknown
                            if debug_mode and (predicted_name != "Unknown" or confidence > 0):
                                y_offset = y2 + 60
                                if predicted_name != "Unknown":
                                    cv2.putText(display_frame, f"ACTUAL: {predicted_name} ({confidence:.2f})", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                                    y_offset += 15
                            
                                for clf_name, result in individual_results.items():
                                    detail_text = f"{clf_name}: {result['name']}({result['confidence']:.2f})"
                                    cv2.putText(display_frame, detail_text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                                    y_offset += 15
        
            # Add system info
            unique_people = len(set(self.known_names)) if self.known_names else 0
            cv2.putText(display_frame, f"Multi-Classifier System | {unique_people} enrolled",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Detected: {detected_count} | Recognized: {recognized_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count} | Threshold: {self.confidence_threshold:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
            if debug_mode:
                cv2.putText(display_frame, "DEBUG MODE ON - Shows actual predictions", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
            # Display frame
            cv2.imshow('Multi-Classifier Attendance', display_frame)
        
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshot_{timestamp}.jpg", display_frame)
                print(f"📸 Screenshot saved: screenshot_{timestamp}.jpg")
            elif key == ord('p'):
                # Print performance report
                print(self.get_classifier_performance_report())
            elif key == ord('d'):
                # Toggle debug mode
                debug_mode = not debug_mode
                print(f"🔍 Debug mode: {'ON' if debug_mode else 'OFF'}")
    
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 SESSION STATISTICS:")
        print(f"Frames processed: {frame_count}")
        print(f"Faces detected: {detected_count}")
        print(f"Faces recognized: {recognized_count}")
        if detected_count > 0:
            print(f"Recognition rate: {(recognized_count/detected_count)*100:.1f}%")
    
        
            
    def generate_report(self, start_date=None, end_date=None):
        """Generate attendance report"""
        return self.database.get_attendance_report(start_date, end_date)
    
    def get_enrolled_users(self):
        """Get enrolled users"""
        return self.database.get_enrolled_users()
    
    def get_system_status(self):
        """Get comprehensive system status"""
        enrolled_users = self.database.get_enrolled_users()
        
        status = {
            'system_type': 'FixedMultiClassifier',
            'enrolled_users': len(enrolled_users),
            'total_embeddings': len(self.known_embeddings),
            'active_classifiers': list(self.classifiers.keys()),
            'confidence_threshold': self.confidence_threshold,
            'verification_threshold': self.verifier.similarity_threshold,
            'classifier_performance': self.classifier_performance
        }
        
        return status
    

# Utility function for easy initialization
def create_multi_classifier_system(model_path='models/', database_path='attendance.db', 
                                   confidence_threshold=0.7, verification_threshold=0.7, device=None):

    return FixedMultiClassifierSystem(
        model_path=model_path,
        database_path=database_path,
        confidence_threshold=confidence_threshold,
        verification_threshold=verification_threshold,
        device=device
    )

if __name__ == "__main__":
    # Example usage
    print("🎯 Multi-Classifier Face Recognition System")
    print("=" * 45)
    
    # Initialize system
    system = create_multi_classifier_system()
    
    print("\n📋 Usage Instructions:")
    print("1. Enroll users: system.enroll_person('John', webcam_capture=True)")
    print("2. Start attendance: system.run_live_attendance()")
    print("3. Check performance: print(system.get_classifier_performance_report())")
    print("4. Generate reports: system.generate_report()")
    
    print(f"\n✅ System ready with {len(system.classifiers)} classifiers!")
    unique_people = len(set(system.known_names)) if system.known_names else 0
    print(f"📊 Currently enrolled: {unique_people} people")