
import cv2
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, date
import sqlite3
from pathlib import Path
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import json

class FaceAttendanceSystem:
    def __init__(self, 
                 model_path='models/',
                 database_path='attendance.db',
                 confidence_threshold=0.7,
                 device=None):
        """
        Initialize Face Recognition Attendance System
        
        Args:
            model_path (str): Path to save/load models
            database_path (str): Path to SQLite database
            confidence_threshold (float): Minimum confidence for recognition
            device: PyTorch device (auto-detected if None)
        """
        
        # Setup paths and parameters
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        
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
            thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
            factor=0.709, 
            post_process=True,
            device=self.device
        )
        
        # FaceNet model for embeddings
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Classifier and label encoder
        self.classifier = None
        self.label_encoder = None
        self.known_embeddings = []
        self.known_names = []
        
        # Initialize database
        self.init_database()
        
        # Load existing model if available
        self.load_model()
    
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
                confidence REAL NOT NULL
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
    
    def enroll_person(self, name, images_path=None, webcam_capture=False, num_samples=10):
        """
        Enroll a new person in the system
        
        Args:
            name (str): Person's name
            images_path (str): Path to folder containing person's images
            webcam_capture (bool): Whether to capture from webcam
            num_samples (int): Number of samples to capture from webcam
        """
        embeddings = []
        
        if webcam_capture:
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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = self.mtcnn.detect(rgb_frame)
                
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
                    embedding = self.extract_face_embedding(frame)
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
            
        elif images_path:
            print(f"Processing images from {images_path}...")
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            
            # Check if images_path contains subfolders for each person
            images_path_obj = Path(images_path)
            person_folder = images_path_obj / name
            
            if person_folder.exists() and person_folder.is_dir():
                # Process images from person's subfolder
                print(f"Looking for images in {person_folder}...")
                for image_file in person_folder.iterdir():
                    if image_file.suffix.lower() in image_extensions:
                        image = cv2.imread(str(image_file))
                        if image is not None:
                            embedding = self.extract_face_embedding(image)
                            if embedding is not None:
                                embeddings.append(embedding)
                                print(f"Processed: {image_file.name}")
                            else:
                                print(f"No face detected in: {image_file.name}")
                        else:
                            print(f"Could not read image: {image_file.name}")
            else:
                # Fallback: process images directly in images_path
                print(f"Person folder {person_folder} not found, checking direct images...")
                for image_file in images_path_obj.iterdir():
                    if image_file.suffix.lower() in image_extensions:
                        image = cv2.imread(str(image_file))
                        if image is not None:
                            embedding = self.extract_face_embedding(image)
                            if embedding is not None:
                                embeddings.append(embedding)
                                print(f"Processed: {image_file.name}")
                            else:
                                print(f"No face detected in: {image_file.name}")
                        else:
                            print(f"Could not read image: {image_file.name}")
        
        if len(embeddings) == 0:
            print(f"No valid face embeddings found for {name}")
            return False
        
        # Add to known embeddings
        self.known_embeddings.extend(embeddings)
        self.known_names.extend([name] * len(embeddings))
        
        # Update database
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO enrolled_users (name, enrollment_date, total_embeddings)
            VALUES (?, ?, ?)
        ''', (name, datetime.now(), len(embeddings)))
        
        conn.commit()
        conn.close()
        
        print(f"Successfully enrolled {name} with {len(embeddings)} samples")
        
        # Retrain classifier
        self.train_classifier()
        self.save_model()
        
        return True
    
    
    def train_classifier(self):
        """Train SVM classifier on known embeddings"""

        if len(self.known_embeddings) < 2:
            print("Need at least 2 embeddings to train classifier")
            return False
        
        # Prepare data
        X = np.array(self.known_embeddings)
        y = np.array(self.known_names)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train SVM
        self.classifier = SVC(
            kernel='linear',
            probability=True,
            C=1.0,
            random_state=42
        )
        self.classifier.fit(X, y_encoded)
        
        print(f"Trained classifier with {len(np.unique(y))} classes")
        return True
    
    
    def predict_face(self, embedding):
        """
        Predict identity from face embedding
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            tuple: (predicted_name, confidence)
        """
        if self.classifier is None or self.label_encoder is None:
            return "Unknown", 0.0
        
        # Reshape for prediction
        embedding = embedding.reshape(1, -1)
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(embedding)[0]
        max_prob_idx = np.argmax(probabilities)
        max_probability = probabilities[max_prob_idx]
        
        # Get predicted class
        predicted_class = self.classifier.predict(embedding)[0]
        predicted_name = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Additional similarity check with known embeddings
        similarities = []
        for known_embedding in self.known_embeddings:
            similarity = cosine_similarity(
                embedding, 
                known_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        max_similarity = max(similarities) if similarities else 0.0
        
        # Combine SVM probability and cosine similarity
        final_confidence = (max_probability + max_similarity) / 2
        
        if final_confidence >= self.confidence_threshold:
            return predicted_name, final_confidence
        else:
            return "Unknown", final_confidence
    
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
    
    def mark_attendance(self, name, confidence):
        """Mark attendance for a person"""
        if self.is_already_marked_today(name):
            return f"Already marked today, {name}!"
        
        # Record attendance
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        cursor.execute('''
            INSERT INTO attendance (name, timestamp, date, confidence)
            VALUES (?, ?, ?, ?)
        ''', (name, now, now.date(), confidence))
        
        conn.commit()
        conn.close()
        
        print(f"Attendance marked for {name} at {now.strftime('%H:%M:%S')}")
        return f"Welcome, {name}!"
    
    def save_model(self):
        """Save trained model and embeddings"""
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'known_embeddings': self.known_embeddings,
            'known_names': self.known_names,
            'confidence_threshold': self.confidence_threshold
        }
        
        with open(self.model_path / 'face_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved successfully")
    
    def load_model(self):
        """Load trained model and embeddings"""
        model_file = self.model_path / 'face_model.pkl'
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.classifier = model_data['classifier']
                self.label_encoder = model_data['label_encoder']
                self.known_embeddings = model_data['known_embeddings']
                self.known_names = model_data['known_names']
                
                print(f"Model loaded successfully with {len(np.unique(self.known_names))} enrolled users")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        
        print("No existing model found")
        return False
    
    def run_attendance_system(self):
        """Run live attendance system with webcam"""
        print("Starting Face Recognition Attendance System...")
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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(rgb_frame)
            
            if boxes is not None:
                for box in boxes:
                    # Draw face box
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Extract face for recognition
                    face_crop = frame[max(0, y1):min(frame.shape[0], y2), 
                                    max(0, x1):min(frame.shape[1], x2)]
                    
                    if face_crop.size > 0:
                        # Get embedding and predict
                        embedding = self.extract_face_embedding(face_crop)
                        
                        if embedding is not None:
                            predicted_name, confidence = self.predict_face(embedding)
                            
                            # Choose color based on recognition
                            if predicted_name != "Unknown":
                                color = (0, 255, 0)  # Green for recognized
                                
                                # Try to mark attendance
                                if confidence >= self.confidence_threshold:
                                    message = self.mark_attendance(predicted_name, confidence)
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
            cv2.putText(display_frame, f"Enrolled: {len(np.unique(self.known_names)) if self.known_names else 0}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Threshold: {self.confidence_threshold:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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
    
    def generate_attendance_report(self, start_date=None, end_date=None):
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


def main():
    parser = argparse.ArgumentParser(description='Face Recognition Attendance System')
    parser.add_argument('--mode', choices=['enroll', 'run', 'report'], 
                       default='run', help='Operation mode')
    # parser.add_argument('--name', type=str, help='Name for enrollment')
    parser.add_argument('--name', nargs='+', help='Name(s) of person(s) to enroll')

    parser.add_argument('--images', type=str, help='Path to images for enrollment')
    parser.add_argument('--webcam', action='store_true', 
                       help='Use webcam for enrollment')
    parser.add_argument('--samples', type=int, default=10, 
                       help='Number of samples for webcam enrollment')
    parser.add_argument('--threshold', type=float, default=0.7, 
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize system
    system = FaceAttendanceSystem(confidence_threshold=args.threshold)
    
    if args.mode == 'enroll':
        if not args.name:
            print("Name is required for enrollment!")
            return
        
        for person_name in args.name:
            system.enroll_person(
                name=person_name,
                images_path=args.images,
                webcam_capture=args.webcam,
                num_samples=args.samples
            )
    
    elif args.mode == 'run':
        system.run_attendance_system()
    
    elif args.mode == 'report':
        df = system.generate_attendance_report()
        print("\nAttendance Report:")
        print(df.to_string(index=False))
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_report_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nReport saved to: {filename}")


if __name__ == "__main__":
    main()