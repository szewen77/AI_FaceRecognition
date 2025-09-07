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
import time
from sklearn.metrics.pairwise import cosine_similarity
# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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
        
        # Check memory usage and optimize if needed
        total_embeddings = len(self.known_embeddings)
        unique_people = len(set(self.known_names))
        avg_per_person = total_embeddings / unique_people if unique_people > 0 else 0
        
        print(f"\n🔄 UPDATING ALL CLASSIFIERS...")
        print(f"📊 Dataset: {total_embeddings} embeddings from {unique_people} people (avg: {avg_per_person:.1f} per person)")
        
        # Memory optimization for large datasets
        if total_embeddings > 1000:
            print("⚠️  Large dataset detected - using memory optimization")
            self._optimize_dataset_for_training()
        
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
                    
                    # Special check for SVM classifier
                    if clf_name == 'SVM' and success and hasattr(classifier, 'svm_model') and classifier.svm_model is not None:
                        if not hasattr(classifier.svm_model, 'support_vectors_') or classifier.svm_model.support_vectors_ is None:
                            print(f"  ⚠️  SVM model may not be properly fitted, attempting to fix...")
                            fix_success = self._fix_svm_classifier(classifier)
                            if fix_success:
                                print(f"  ✅ SVM model fixed successfully")
                            else:
                                print(f"  ⚠️  SVM model fix failed, will use similarity-based fallback")
                
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
    
    def _optimize_dataset_for_training(self, max_images_per_person=10):
        """Optimize dataset for training by limiting images per person"""
        if len(self.known_embeddings) == 0:
            return
        
        # Group embeddings by person
        person_embeddings = {}
        for embedding, name in zip(self.known_embeddings, self.known_names):
            if name not in person_embeddings:
                person_embeddings[name] = []
            person_embeddings[name].append(embedding)
        
        # Optimize each person's embeddings
        optimized_embeddings = []
        optimized_names = []
        
        for name, embeddings in person_embeddings.items():
            if len(embeddings) > max_images_per_person:
                # Take a diverse sample
                import random
                random.seed(42)
                selected_embeddings = random.sample(embeddings, max_images_per_person)
                print(f"  📊 {name}: {len(embeddings)} → {len(selected_embeddings)} images")
            else:
                selected_embeddings = embeddings
            
            optimized_embeddings.extend(selected_embeddings)
            optimized_names.extend([name] * len(selected_embeddings))
        
        # Update the dataset
        self.known_embeddings = optimized_embeddings
        self.known_names = optimized_names
        
        print(f"✅ Dataset optimized: {len(self.known_embeddings)} embeddings from {len(set(self.known_names))} people")
    
    def _fix_svm_classifier(self, svm_classifier):
        """Fix SVM classifier if it's not properly trained"""
        try:
            from sklearn.svm import SVC
            import numpy as np
            
            # Get current data
            X = np.array(svm_classifier.known_embeddings)
            y = np.array(svm_classifier.known_names)
            
            # Encode labels
            y_encoded = svm_classifier.label_encoder.transform(y)
            
            # Try different SVM configurations
            configs = [
                {'kernel': 'linear', 'C': 1.0},
                {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
                {'kernel': 'rbf', 'C': 0.1, 'gamma': 'auto'},
                {'kernel': 'poly', 'C': 1.0, 'degree': 2}
            ]
            
            for i, config in enumerate(configs):
                try:
                    print(f"🔄 Trying SVM config {i+1}: {config}")
                    svm_classifier.svm_model = SVC(
                        probability=True,
                        random_state=42,
                        **config
                    )
                    svm_classifier.svm_model.fit(X, y_encoded)
                    
                    # Check if properly fitted
                    if hasattr(svm_classifier.svm_model, 'support_vectors_') and svm_classifier.svm_model.support_vectors_ is not None:
                        print(f"✅ SVM fixed with config {i+1}: {config}")
                        return True
                        
                except Exception as e:
                    print(f"❌ SVM config {i+1} failed: {e}")
                    continue
            
            print("❌ All SVM configurations failed, using similarity-based fallback")
            return False
            
        except Exception as e:
            print(f"❌ Error fixing SVM classifier: {e}")
            return False
    
    def train_test_evaluate_and_save(self, test_size=0.2, random_state=42, output_to_files=True):
        """
        Perform a stratified train/test split, train on train set only, evaluate on test set,
        and optionally save detailed metrics for each classifier.
        """
        if len(self.known_embeddings) < 10:
            print("Need at least 10 samples to run train/test evaluation")
            return {"error": "insufficient_data"}

        # Prepare data
        X = np.array(self.known_embeddings)
        y = np.array(self.known_names)

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Try stratified split; fall back to regular split if needed
        try:
            unique, counts = np.unique(y_encoded, return_counts=True)
            min_samples_per_class = int(min(counts)) if len(counts) > 0 else 0
            if min_samples_per_class >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                )
            else:
                print("⚠️  Some classes have <2 samples. Using non-stratified split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state
                )
        except ValueError as e:
            print(f"⚠️  Stratified split failed: {e}. Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state
            )

        results = {}
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Train and evaluate each classifier
        for clf_name, clf in self.classifiers.items():
            print(f"\n🔄 Training {clf_name} on train split and evaluating on test split...")

            # Train
            start_time = time.time()
            clf.train(X_train, le.inverse_transform(y_train))
            training_time = time.time() - start_time

            # Predict on test
            start_time = time.time()
            y_pred_encoded = []
            y_pred_conf = []
            for sample in X_test:
                try:
                    pred_name, confidence = clf.predict(sample)
                    pred_encoded = le.transform([pred_name])[0] if pred_name in le.classes_ else np.bincount(y_train).argmax()
                except Exception:
                    pred_encoded = np.bincount(y_train).argmax()
                    confidence = 0.0
                y_pred_encoded.append(pred_encoded)
                y_pred_conf.append(float(confidence))

            inference_time = time.time() - start_time

            y_pred_encoded = np.array(y_pred_encoded)

            # Metrics
            acc = accuracy_score(y_test, y_pred_encoded)
            prec = precision_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred_encoded)

            results[clf_name] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist(),
                'class_names': le.classes_.tolist(),
                'training_time': float(training_time),
                'inference_time': float(inference_time),
                'inference_speed': float(len(X_test) / inference_time) if inference_time > 0 else 0.0,
            }

            print(f"  ✅ {clf_name} | Acc: {acc:.3f} | F1: {f1:.3f}")

        # Optionally save results
        if output_to_files:
            try:
                import json
                self.model_path.mkdir(parents=True, exist_ok=True)

                # Per-classifier JSON
                for clf_name, metrics in results.items():
                    out_path = self.model_path / f"{clf_name}_results.json"
                    with open(out_path, 'w') as f:
                        json.dump(metrics, f, indent=2)

                # Combined CSV
                comparison_rows = []
                for clf_name, m in results.items():
                    comparison_rows.append({
                        'classifier': clf_name,
                        'accuracy': m['accuracy'],
                        'precision': m['precision'],
                        'recall': m['recall'],
                        'f1_score': m['f1_score'],
                        'training_time': m['training_time'],
                        'inference_time': m['inference_time'],
                        'inference_speed': m['inference_speed']
                    })
                df = pd.DataFrame(comparison_rows)
                csv_path = self.model_path / f"algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_path, index=False)

                # Comprehensive results with metadata
                training_metadata = {
                    'dataset_path': str(self.model_path.parent / 'Original Images'),
                    'total_embeddings': int(len(self.known_embeddings)),
                    'total_people': int(len(set(self.known_names))),
                    'max_images_per_person': int(pd.Series(self.known_names).value_counts().max()) if self.known_names else 0,
                    'min_images_per_person': int(pd.Series(self.known_names).value_counts().min()) if self.known_names else 0,
                    'training_date': timestamp,
                    'person_names': sorted(list(set(self.known_names)))
                }

                comprehensive = {
                    'training_metadata': training_metadata,
                    'evaluation_results': results
                }
                with open(self.model_path / 'comprehensive_evaluation_results.json', 'w') as f:
                    json.dump(comprehensive, f, indent=2)

                print("💾 Saved evaluation artifacts to 'models/'")
            except Exception as e:
                print(f"⚠️  Failed to save evaluation artifacts: {e}")

        return results

    def bulk_enroll_from_directories_and_evaluate(self, faces_dir='Faces/Faces', originals_dir='Original Images/Original Images', max_images_per_person=None, test_size=0.2):
        """
        Scan Faces and Original Images directories, enroll all users, retrain, and evaluate.
        - faces_dir: path where each person's images are stored (flat with name in filename or per-folder)
        - originals_dir: optional second dataset to include
        - max_images_per_person: cap per person if desired
        - test_size: fraction for held-out test split
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        def iter_people_images(root_dir):
            root = Path(root_dir)
            if not root.exists():
                return {}
            people_to_files = {}
            # Case 1: per-person subfolders
            for sub in root.iterdir():
                if sub.is_dir():
                    person_name = sub.name
                    files = [p for p in sub.iterdir() if p.suffix.lower() in image_extensions]
                    if files:
                        people_to_files.setdefault(person_name, []).extend(files)
            # Case 2: flat folder, names embedded in filenames like `Name_XX.jpg`
            flat_files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
            for f in flat_files:
                base = f.stem
                # Split on last underscore if present
                if '_' in base:
                    person_name = base.rsplit('_', 1)[0]
                else:
                    person_name = base
                people_to_files.setdefault(person_name, []).append(f)
            return people_to_files

        combined = {}
        for directory in [faces_dir, originals_dir]:
            ppl = iter_people_images(directory)
            for name, files in ppl.items():
                if max_images_per_person:
                    files = files[:max_images_per_person]
                combined.setdefault(name, [])
                # Deduplicate while preserving order
                seen = set()
                for fp in files:
                    s = str(fp)
                    if s not in seen:
                        combined[name].append(fp)
                        seen.add(s)

        # Enroll everyone
        enrolled_count = 0
        for name, files in combined.items():
            if not files:
                continue
            print(f"\n=== Enrolling {name} ({len(files)} images) ===")
            self.enrollment_system.enroll_from_file_list(name, [str(p) for p in files])
            enrolled_count += 1

        print(f"\n✅ Enrollment complete. People processed: {enrolled_count}")

        # Rebuild embeddings and retrain classifiers with all data
        self._rebuild_embeddings_from_database()
        self._update_all_classifiers_with_new_data()
        self.save_system()

        # Evaluate with held-out test set and save metrics
        return self.train_test_evaluate_and_save(test_size=test_size, random_state=42, output_to_files=True)

    def enroll_from_single_dataset_root_and_evaluate(self, dataset_root, max_images_per_person=None, test_size=0.2):
        """
        Support a single folder with subfolders per person, e.g.:
            dataset/
              person1/*.jpg
              person2/*.jpg
        Enrolls everyone found, retrains, and evaluates.
        """
        root = Path(dataset_root)
        if not root.exists():
            print(f"Dataset root not found: {dataset_root}")
            return {"error": "dataset_not_found"}

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        people = {}
        for sub in root.iterdir():
            if sub.is_dir():
                files = [p for p in sub.iterdir() if p.suffix.lower() in image_extensions]
                if max_images_per_person:
                    files = files[:max_images_per_person]
                if files:
                    people[sub.name] = files

        if not people:
            print("No people/images found under dataset root")
            return {"error": "no_data"}

        enrolled = 0
        for name, files in people.items():
            print(f"\n=== Enrolling {name} ({len(files)} images) ===")
            self.enrollment_system.enroll_from_file_list(name, [str(p) for p in files])
            enrolled += 1

        print(f"\n✅ Enrollment complete from single root. People processed: {enrolled}")

        # Rebuild, retrain, save, evaluate
        self._rebuild_embeddings_from_database()
        self._update_all_classifiers_with_new_data()
        self.save_system()
        return self.train_test_evaluate_and_save(test_size=test_size, random_state=42, output_to_files=True)
    
    def _enroll_from_directory(self, directory_path, prefix="", max_samples_per_person=10, existing_names=None):
        """Helper method to enroll people from a directory with limited samples per person and duplicate checking"""
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return 0
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        people_count = 0
        
        # Check if it's a flat directory with named files
        flat_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        if flat_files:
            # Group by person name (extracted from filename)
            people_files = {}
            for f in flat_files:
                base = f.stem
                if '_' in base:
                    person_name = base.rsplit('_', 1)[0]
                else:
                    person_name = base
                
                if prefix:
                    person_name = f"{prefix}{person_name}"
                
                if person_name not in people_files:
                    people_files[person_name] = []
                people_files[person_name].append(f)
            
            # Enroll each person with limited samples
            for person_name, files in people_files.items():
                # Check for duplicates if existing_names is provided
                if existing_names is not None and person_name in existing_names:
                    print(f"⏭️  Skipping {person_name}: already exists")
                    continue
                
                try:
                    # Limit to max_samples_per_person
                    original_count = len(files)
                    if len(files) > max_samples_per_person:
                        import random
                        files = random.sample(files, max_samples_per_person)
                        print(f"📊 {person_name}: {len(files)} samples (limited from {original_count})")
                    else:
                        print(f"📊 {person_name}: {len(files)} samples")
                    
                    self.enrollment_system.enroll_from_file_list(person_name, [str(f) for f in files])
                    people_count += 1
                    
                    # Add to existing_names to avoid duplicates in same run
                    if existing_names is not None:
                        existing_names.add(person_name)
                        
                except Exception as e:
                    print(f"⚠️  Error enrolling {person_name}: {e}")
        
        # Check if it's a directory with subdirectories per person
        else:
            for subdir in dir_path.iterdir():
                if subdir.is_dir():
                    files = [f for f in subdir.iterdir() if f.suffix.lower() in image_extensions]
                    if files:
                        person_name = f"{prefix}{subdir.name}" if prefix else subdir.name
                        
                        # Check for duplicates if existing_names is provided
                        if existing_names is not None and person_name in existing_names:
                            print(f"⏭️  Skipping {person_name}: already exists")
                            continue
                        
                        try:
                            # Limit to max_samples_per_person
                            original_count = len(files)
                            if len(files) > max_samples_per_person:
                                import random
                                files = random.sample(files, max_samples_per_person)
                                print(f"📊 {person_name}: {len(files)} samples (limited from {original_count})")
                            else:
                                print(f"📊 {person_name}: {len(files)} samples")
                            
                            self.enrollment_system.enroll_from_file_list(person_name, [str(f) for f in files])
                            people_count += 1
                            
                            # Add to existing_names to avoid duplicates in same run
                            if existing_names is not None:
                                existing_names.add(person_name)
                                
                        except Exception as e:
                            print(f"⚠️  Error enrolling {person_name}: {e}")
        
        return people_count
    
    def add_more_students(self, add_count=100, faces_dir='Faces/Faces', 
                         originals_dir='Original Images/Original Images',
                         lfw_path='lfw-funneled/lfw_funneled', max_lfw_people=50):
        """
        Add more students from datasets (default 100 more students)
        
        Args:
            add_count: Number of additional students to add
            faces_dir: Path to Faces dataset
            originals_dir: Path to Original Images dataset  
            lfw_path: Path to LFW dataset
            max_lfw_people: Maximum LFW people to add
        """
        # Get current count and existing names
        current_users = self.database.get_enrolled_users()
        current_count = len(current_users)
        existing_names = set(current_users['name'].tolist())
        
        print(f"📊 Current students: {current_count}")
        print(f"🎯 Adding {add_count} more students")
        print(f"📈 Final target: {current_count + add_count} students")
        
        needed = add_count
        print(f"📈 Need to add: {needed} more students")
        
        added_count = 0
        
        # 1. Try to add from Original Images dataset first (your custom data)
        if needed > 0:
            print(f"\n1️⃣ Adding from Original Images dataset...")
            originals_added = self._enroll_from_directory(originals_dir, prefix="Student_", 
                                                        max_samples_per_person=10, 
                                                        existing_names=existing_names)
            added_count += originals_added
            needed -= originals_added
            print(f"✅ Added {originals_added} students from Original Images")
        
        # 2. Try to add from Faces dataset
        if needed > 0:
            print(f"\n2️⃣ Adding from Faces dataset...")
            faces_added = self._enroll_from_directory(faces_dir, prefix="Student_", 
                                                    max_samples_per_person=10,
                                                    existing_names=existing_names)
            added_count += faces_added
            needed -= faces_added
            print(f"✅ Added {faces_added} students from Faces")
        
        # 3. Add from LFW dataset if still needed
        if needed > 0:
            print(f"\n3️⃣ Adding from LFW dataset...")
            lfw_added = self._enroll_from_lfw_subset(lfw_path, max_people=min(needed, max_lfw_people),
                                                   max_samples_per_person=10, existing_names=existing_names)
            added_count += lfw_added
            needed -= lfw_added
            print(f"✅ Added {lfw_added} students from LFW")
        
        # Rebuild embeddings and retrain
        if added_count > 0:
            print(f"\n🔄 Rebuilding embeddings and retraining...")
            self._rebuild_embeddings_from_database()
            self._update_all_classifiers_with_new_data()
            self.save_system()
        
        final_count = len(self.database.get_enrolled_users())
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"📊 Total students: {final_count}")
        print(f"📈 Students added: {added_count}")
        print(f"✅ Target reached: {'Yes' if added_count >= add_count else 'No'}")
        
        return {
            "current": current_count,
            "added": added_count,
            "final": final_count,
            "target_reached": added_count >= add_count
        }
    
    def _enroll_from_lfw_subset(self, lfw_path, max_people=50, max_samples_per_person=10, existing_names=None):
        """Enroll a subset of people from LFW dataset with limited samples per person and duplicate checking"""
        lfw_dir = Path(lfw_path)
        if not lfw_dir.exists():
            print(f"LFW dataset not found at: {lfw_path}")
            return 0
        
        # Get all person directories
        person_dirs = [d for d in lfw_dir.iterdir() if d.is_dir()]
        
        # Limit to max_people
        person_dirs = person_dirs[:max_people]
        
        enrolled_count = 0
        
        for person_dir in person_dirs:
            person_name = f"Student_LFW_{person_dir.name}"
            
            # Check for duplicates if existing_names is provided
            if existing_names is not None and person_name in existing_names:
                print(f"⏭️  Skipping {person_name}: already exists")
                continue
            
            # Get all images for this person
            all_image_files = [f for f in person_dir.iterdir() 
                             if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            
            if len(all_image_files) == 0:
                continue
            
            # Limit to max_samples_per_person
            if len(all_image_files) > max_samples_per_person:
                import random
                image_files = random.sample(all_image_files, max_samples_per_person)
                print(f"📊 {person_name}: {len(image_files)} samples (limited from {len(all_image_files)})")
            else:
                image_files = all_image_files
                print(f"📊 {person_name}: {len(image_files)} samples")
            
            try:
                self.enrollment_system.enroll_from_file_list(person_name, [str(f) for f in image_files])
                enrolled_count += 1
                
                # Add to existing_names to avoid duplicates in same run
                if existing_names is not None:
                    existing_names.add(person_name)
                
                if enrolled_count % 10 == 0:
                    print(f"📸 Processed {enrolled_count} LFW students...")
                    
            except Exception as e:
                print(f"⚠️  Error processing {person_name}: {e}")
                continue
        
        return enrolled_count

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
    
    def comprehensive_classifier_evaluation(self, test_size=0.3, cv_folds=5, save_plots=True, use_saved_results=True):
        """
        Enhanced comprehensive evaluation that can use pre-computed training results
        """
        # Check if we have saved comprehensive results
        results_path = self.model_path / "comprehensive_evaluation_results.json"
        
        if use_saved_results and results_path.exists():
            print("📊 Loading saved comprehensive evaluation results...")
            try:
                import json
                with open(results_path, 'r') as f:
                    saved_data = json.load(f)
                
                # Check if results are recent and valid
                evaluation_results = saved_data.get('evaluation_results', {})
                training_metadata = saved_data.get('training_metadata', {})
                
                if evaluation_results and len(evaluation_results) >= 3:  # All three classifiers
                    print("✅ Using saved comprehensive evaluation results")
                    print(f"📅 Training Date: {training_metadata.get('training_date', 'Unknown')}")
                    print(f"👥 Dataset Size: {training_metadata.get('total_people', 0)} people, {training_metadata.get('total_embeddings', 0)} embeddings")
                    
                    # Convert back to the expected format
                    results = {}
                    for clf_name, metrics in evaluation_results.items():
                        # Convert confusion matrix back to numpy array if it's a list
                        if 'confusion_matrix' in metrics and isinstance(metrics['confusion_matrix'], list):
                            metrics['confusion_matrix'] = np.array(metrics['confusion_matrix'])
                        
                        results[clf_name] = metrics
                    
                    # Add training metadata to results for display
                    results['training_metadata'] = training_metadata
                    
                    # Generate enhanced comparison report
                    comparison_report = self._generate_enhanced_comparison_report(results, training_metadata)
                    results['comparison_report'] = comparison_report
                    
                    # Note: Plotting functionality removed to avoid threading issues
                    
                    return results
                
            except Exception as e:
                print(f"⚠️  Error loading saved results: {e}")
                print("Falling back to real-time evaluation...")
        
        # Fall back to original evaluation if no saved results or if requested
        print("🔄 Running real-time classifier evaluation...")
        
        if len(self.known_embeddings) < 10:
            return {"error": "Need at least 10 samples for comprehensive evaluation"}
        
        print("🔄 Starting comprehensive classifier evaluation...")
        results = {}
        
        # Prepare data
        X = np.array(self.known_embeddings)
        y = np.array(self.known_names)
        
        # Get unique classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            return {"error": "Need at least 2 classes for evaluation"}
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data with handling for small classes
        try:
            # Check if we can use stratified split
            unique, counts = np.unique(y_encoded, return_counts=True)
            min_samples_per_class = min(counts)
            
            if min_samples_per_class >= 2:
                # Use stratified split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
            else:
                # Some classes have only 1 sample, use regular split
                print("⚠️  Some classes have only 1 sample. Using regular (non-stratified) split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42
                )
        except ValueError as e:
            print(f"⚠️  Stratified split failed: {e}. Using regular split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42
            )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        print(f"📊 Classes: {n_classes} ({unique_classes})")
        
        # Evaluate each classifier
        for clf_name, clf in self.classifiers.items():
            print(f"\n🔄 Evaluating {clf_name}...")
            
            # Train classifier
            start_time = time.time()
            clf.train(X_train, le.inverse_transform(y_train))
            training_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred_encoded = []
            y_pred_proba = []
            
            for sample in X_test:
                try:
                    pred_result = clf.predict([sample])
                    if isinstance(pred_result[0], tuple):
                        pred_name, confidence = pred_result[0]
                    else:
                        pred_name = pred_result[0]
                        confidence = 1.0
                    
                    # Encode prediction
                    try:
                        pred_encoded = le.transform([pred_name])[0]
                    except ValueError:
                        # Unknown class, assign to most frequent class
                        pred_encoded = np.bincount(y_train).argmax()
                    
                    y_pred_encoded.append(pred_encoded)
                    y_pred_proba.append(confidence)
                except:
                    # If prediction fails, assign to most frequent class
                    y_pred_encoded.append(np.bincount(y_train).argmax())
                    y_pred_proba.append(0.0)
            
            inference_time = time.time() - start_time
            
            y_pred_encoded = np.array(y_pred_encoded)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred_encoded)
            precision = precision_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_encoded)
            
            # Cross-validation with proper handling of small datasets
            try:
                # Check minimum samples per class
                unique, counts = np.unique(y_encoded, return_counts=True)
                min_samples_per_class = min(counts)
                
                if min_samples_per_class >= 2:
                    # Use stratified k-fold with adjusted CV folds
                    effective_cv_folds = min(cv_folds, min_samples_per_class, len(X)//3)
                    if effective_cv_folds >= 2:
                        cv_scores = cross_val_score(
                            clf, X, y_encoded, cv=effective_cv_folds, 
                            scoring='accuracy'
                        )
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    else:
                        # Not enough data for proper CV
                        cv_mean = accuracy
                        cv_std = 0.0
                else:
                    # Some classes have only 1 sample, can't do stratified CV
                    cv_mean = accuracy
                    cv_std = 0.0
            except Exception as cv_error:
                print(f"⚠️  Cross-validation failed: {cv_error}")
                cv_mean = accuracy
                cv_std = 0.0
            
            # Per-class metrics
            precision_per_class = precision_score(y_test, y_pred_encoded, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred_encoded, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred_encoded, average=None, zero_division=0)
            
            # Store results
            results[clf_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std,
                'training_time': training_time,
                'inference_time': inference_time,
                'inference_speed': len(X_test) / inference_time,  # samples per second
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class,
                'class_names': le.classes_,
                'y_test': y_test,
                'y_pred': y_pred_encoded,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  ✅ Accuracy: {accuracy:.3f}")
            print(f"  ✅ Precision: {precision:.3f}")
            print(f"  ✅ Recall: {recall:.3f}")
            print(f"  ✅ F1-Score: {f1:.3f}")
            print(f"  ⏱️ Training time: {training_time:.3f}s")
            print(f"  ⚡ Inference speed: {len(X_test) / inference_time:.1f} samples/sec")
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(results)
        results['comparison_report'] = comparison_report
        
        # Note: Plotting functionality removed to avoid threading issues
        
        print("\n✅ Comprehensive evaluation completed!")
        return results
    
    def _generate_comparison_report(self, results):
        """Generate a detailed comparison report"""
        report = "=" * 80 + "\n"
        report += "                    COMPREHENSIVE CLASSIFIER COMPARISON\n"
        report += "=" * 80 + "\n\n"
        
        # Performance metrics table
        report += "📊 PERFORMANCE METRICS COMPARISON:\n"
        report += "-" * 80 + "\n"
        report += f"{'Classifier':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
        report += "-" * 80 + "\n"
        
        for clf_name, metrics in results.items():
            if clf_name == 'comparison_report':
                continue
            report += f"{clf_name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
            report += f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}\n"
        
        # Cross-validation results
        report += "\n📈 CROSS-VALIDATION RESULTS:\n"
        report += "-" * 60 + "\n"
        report += f"{'Classifier':<20} {'CV Accuracy':<15} {'Std Dev':<10}\n"
        report += "-" * 60 + "\n"
        
        for clf_name, metrics in results.items():
            if clf_name == 'comparison_report':
                continue
            report += f"{clf_name:<20} {metrics['cv_accuracy_mean']:<15.3f} {metrics['cv_accuracy_std']:<10.3f}\n"
        
        # Computational efficiency
        report += "\n⚡ COMPUTATIONAL EFFICIENCY:\n"
        report += "-" * 70 + "\n"
        report += f"{'Classifier':<20} {'Training(s)':<12} {'Inference(s)':<12} {'Speed(sps)':<12}\n"
        report += "-" * 70 + "\n"
        
        for clf_name, metrics in results.items():
            if clf_name == 'comparison_report':
                continue
            report += f"{clf_name:<20} {metrics['training_time']:<12.3f} "
            report += f"{metrics['inference_time']:<12.3f} {metrics['inference_speed']:<12.1f}\n"
        
        # Best performer in each category
        report += "\n🏆 BEST PERFORMERS:\n"
        report += "-" * 40 + "\n"
        
        # Find best in each metric
        best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'] if x[0] != 'comparison_report' else 0)
        best_precision = max(results.items(), key=lambda x: x[1]['precision'] if x[0] != 'comparison_report' else 0)
        best_recall = max(results.items(), key=lambda x: x[1]['recall'] if x[0] != 'comparison_report' else 0)
        best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'] if x[0] != 'comparison_report' else 0)
        best_speed = max(results.items(), key=lambda x: x[1]['inference_speed'] if x[0] != 'comparison_report' else 0)
        
        report += f"🎯 Best Accuracy:   {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.3f})\n"
        report += f"🎯 Best Precision:  {best_precision[0]} ({best_precision[1]['precision']:.3f})\n"
        report += f"🎯 Best Recall:     {best_recall[0]} ({best_recall[1]['recall']:.3f})\n"
        report += f"🎯 Best F1-Score:   {best_f1[0]} ({best_f1[1]['f1_score']:.3f})\n"
        report += f"⚡ Fastest:         {best_speed[0]} ({best_speed[1]['inference_speed']:.1f} sps)\n"
        
        # Recommendations
        report += "\n💡 RECOMMENDATIONS:\n"
        report += "-" * 50 + "\n"
        
        if best_accuracy[1]['accuracy'] > 0.9:
            report += f"✅ For maximum accuracy: Use {best_accuracy[0]}\n"
        
        if best_speed[1]['inference_speed'] > 100:
            report += f"⚡ For real-time applications: Use {best_speed[0]}\n"
        
        if best_f1[1]['f1_score'] > 0.85:
            report += f"⚖️  For balanced performance: Use {best_f1[0]}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    def _generate_enhanced_comparison_report(self, results, training_metadata):
        """Generate enhanced comparison report with training metadata"""
        report = "=" * 80 + "\n"
        report += "                    ENHANCED ALGORITHM COMPARISON REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Training dataset information
        report += f"📊 TRAINING DATASET INFORMATION:\n"
        report += f"  • Dataset Path: {training_metadata.get('dataset_path', 'Unknown')}\n"
        report += f"  • Total People: {training_metadata.get('total_people', 0)}\n"
        report += f"  • Total Embeddings: {training_metadata.get('total_embeddings', 0)}\n"
        report += f"  • Training Date: {training_metadata.get('training_date', 'Unknown')}\n"
        report += f"  • Max Images per Person: {training_metadata.get('max_images_per_person', 'Unknown')}\n\n"
        
        # Filter out metadata from results for metrics calculation
        metrics_results = {k: v for k, v in results.items() if k != 'training_metadata'}
        
        # Performance metrics table
        report += f"📊 PERFORMANCE METRICS COMPARISON:\n"
        report += "-" * 80 + "\n"
        report += f"{'Algorithm':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
        report += "-" * 80 + "\n"
        
        for clf_name, metrics in metrics_results.items():
            report += f"{clf_name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
            report += f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}\n"
        
        # Computational efficiency
        report += f"\n⚡ COMPUTATIONAL EFFICIENCY:\n"
        report += "-" * 70 + "\n"
        report += f"{'Algorithm':<20} {'Training(s)':<12} {'Inference(s)':<12} {'Speed(sps)':<12}\n"
        report += "-" * 70 + "\n"
        
        for clf_name, metrics in metrics_results.items():
            report += f"{clf_name:<20} {metrics.get('training_time', 0):<12.3f} "
            report += f"{metrics.get('inference_time', 0):<12.3f} {metrics.get('inference_speed', 0):<12.1f}\n"
        
        # Best performers
        report += f"\n🏆 BEST PERFORMERS:\n"
        report += "-" * 40 + "\n"
        
        # Find best in each metric
        best_accuracy = max(metrics_results.items(), key=lambda x: x[1]['accuracy'])
        best_precision = max(metrics_results.items(), key=lambda x: x[1]['precision'])
        best_recall = max(metrics_results.items(), key=lambda x: x[1]['recall'])
        best_f1 = max(metrics_results.items(), key=lambda x: x[1]['f1_score'])
        best_speed = max(metrics_results.items(), key=lambda x: x[1].get('inference_speed', 0))
        
        report += f"🎯 Best Accuracy:   {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.3f})\n"
        report += f"🎯 Best Precision:  {best_precision[0]} ({best_precision[1]['precision']:.3f})\n"
        report += f"🎯 Best Recall:     {best_recall[0]} ({best_recall[1]['recall']:.3f})\n"
        report += f"🎯 Best F1-Score:   {best_f1[0]} ({best_f1[1]['f1_score']:.3f})\n"
        report += f"⚡ Fastest:         {best_speed[0]} ({best_speed[1].get('inference_speed', 0):.1f} sps)\n"
        
        # Recommendations
        report += f"\n💡 RECOMMENDATIONS:\n"
        report += "-" * 50 + "\n"
        
        if best_accuracy[1]['accuracy'] > 0.9:
            report += f"✅ For maximum accuracy: Use {best_accuracy[0]}\n"
        
        if best_speed[1].get('inference_speed', 0) > 10:
            report += f"⚡ For real-time applications: Use {best_speed[0]}\n"
        
        if best_f1[1]['f1_score'] > 0.85:
            report += f"⚖️  For balanced performance: Use {best_f1[0]}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    # Enhanced plotting function removed to avoid threading issues
    
    # Plotting function removed to avoid threading issues
    
    def _rebuild_embeddings_from_database(self, max_images_per_person=10):
        """Rebuild embeddings and names from database with memory optimization"""
        self.known_embeddings = []
        self.known_names = []
        
        try:
            enrolled_users = self.database.get_enrolled_users()
            total_users = len(enrolled_users)
            
            print(f"🔄 Rebuilding embeddings from {total_users} users...")
            
            for idx, (_, user) in enumerate(enrolled_users.iterrows()):
                name = user['name']
                embeddings = self.database.get_user_embeddings(name)
                
                # Limit images per person to prevent memory issues
                if len(embeddings) > max_images_per_person:
                    print(f"  📊 {name}: {len(embeddings)} images → limiting to {max_images_per_person}")
                    # Take a random sample to maintain diversity
                    import random
                    random.seed(42)  # For reproducibility
                    embeddings = random.sample(embeddings, max_images_per_person)
                else:
                    print(f"  📊 {name}: {len(embeddings)} images")
                
                self.known_embeddings.extend(embeddings)
                self.known_names.extend([name] * len(embeddings))
                
                # Progress indicator for large datasets
                if total_users > 20 and (idx + 1) % 10 == 0:
                    print(f"  ✅ Processed {idx + 1}/{total_users} users...")
            
            print(f"✅ Rebuilt embeddings: {len(self.known_embeddings)} total from {len(set(self.known_names))} users")
                
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
        
        # Calculate detailed statistics
        total_embeddings_in_db = 0
        max_images_per_person = 0
        min_images_per_person = float('inf')
        
        for _, user in enrolled_users.iterrows():
            embeddings = self.database.get_user_embeddings(user['name'])
            total_embeddings_in_db += len(embeddings)
            max_images_per_person = max(max_images_per_person, len(embeddings))
            min_images_per_person = min(min_images_per_person, len(embeddings))
        
        if min_images_per_person == float('inf'):
            min_images_per_person = 0
        
        avg_images_per_person = total_embeddings_in_db / len(enrolled_users) if len(enrolled_users) > 0 else 0
        
        status = {
            'system_type': 'FixedMultiClassifier',
            'enrolled_users': len(enrolled_users),
            'total_embeddings': len(self.known_embeddings),
            'total_embeddings_in_db': total_embeddings_in_db,
            'avg_images_per_person': round(avg_images_per_person, 1),
            'max_images_per_person': max_images_per_person,
            'min_images_per_person': min_images_per_person,
            'active_classifiers': list(self.classifiers.keys()),
            'confidence_threshold': self.confidence_threshold,
            'verification_threshold': self.verifier.similarity_threshold,
            'classifier_performance': self.classifier_performance,
            'memory_optimized': len(self.known_embeddings) < total_embeddings_in_db
        }
        
        return status
    
    def remove_students_with_insufficient_samples(self, min_samples=10):
        """Remove students who have less than the specified minimum number of samples"""
        try:
            enrolled_users = self.database.get_enrolled_users()
            removed_students = []
            kept_students = []
            
            print(f"Checking students for minimum {min_samples} samples...")
            
            for _, user in enrolled_users.iterrows():
                student_name = user['name']
                embeddings = self.database.get_user_embeddings(student_name)
                sample_count = len(embeddings)
                
                if sample_count < min_samples:
                    print(f"Removing {student_name}: {sample_count} samples (less than {min_samples})")
                    # Remove from database
                    success, message = self.database.delete_enrolled_user(student_name)
                    if success:
                        removed_students.append({
                            'name': student_name,
                            'samples': sample_count
                        })
                        print(f"✅ Successfully removed {student_name}")
                    else:
                        print(f"❌ Failed to remove {student_name}: {message}")
                else:
                    print(f"Keeping {student_name}: {sample_count} samples")
                    kept_students.append({
                        'name': student_name,
                        'samples': sample_count
                    })
            
            # Rebuild embeddings and retrain classifiers if any students were removed
            if removed_students:
                print(f"\nRemoved {len(removed_students)} students with insufficient samples.")
                print("Rebuilding embeddings and retraining classifiers...")
                
                # Rebuild embeddings from database
                self._rebuild_embeddings_from_database(max_images_per_person=10)
                
                # Retrain all classifiers
                self._update_all_classifiers_with_new_data()
                
                # Save updated system
                self.save_system()
                
                print("System updated successfully!")
            else:
                print("No students removed - all have sufficient samples.")
            
            return {
                'removed_count': len(removed_students),
                'kept_count': len(kept_students),
                'removed_students': removed_students,
                'kept_students': kept_students,
                'min_samples': min_samples
            }
            
        except Exception as e:
            error_msg = f"Error removing students with insufficient samples: {str(e)}"
            print(error_msg)
            return {
                'error': error_msg,
                'removed_count': 0,
                'kept_count': 0,
                'removed_students': [],
                'kept_students': []
            }
    

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