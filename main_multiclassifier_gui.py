# main_multiclassifier_gui.py - GUI Face Recognition with Multiple Classifiers
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import sys
import cv2
from pathlib import Path
from datetime import datetime
import queue
import time

# Add classifiers directory to path
sys.path.append('classifiers')

from face_core import FaceRecognitionSystem
from svm_classifier import SVMClassifier
from knn_classifier import KNNClassifier
from logistic_regression import LogisticRegressionClassifier

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # System variables
        self.system = None
        self.current_classifier = "KNN"
        self.camera_running = False
        self.cap = None
        self.log_queue = queue.Queue()
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize system with default classifier
        self.initialize_system()
        
        # Start log processing
        self.process_log_queue()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🎓 Face Recognition Attendance System", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Controls
        left_frame = tk.LabelFrame(main_frame, text="Controls", font=('Arial', 12, 'bold'), 
                                  bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        
        # Classifier selection
        classifier_frame = tk.LabelFrame(left_frame, text="Select Classifier", 
                                       font=('Arial', 10, 'bold'), bg='#f0f0f0')
        classifier_frame.pack(fill='x', pady=(0, 15))
        
        self.classifier_var = tk.StringVar(value="KNN")
        
        # Classifier radio buttons with descriptions
        classifiers = [
            ("KNN", "K-Nearest Neighbors\n• Fast training\n• Good for small datasets\n• Instance-based learning"),
            ("SVM", "Support Vector Machine\n• Good generalization\n• Effective in high dimensions\n• Memory efficient"),
            ("LogisticRegression", "Logistic Regression\n• Probabilistic output\n• Linear decision boundary\n• Simple and interpretable")
        ]
        
        for classifier, description in classifiers:
            frame = tk.Frame(classifier_frame, bg='#f0f0f0')
            frame.pack(fill='x', pady=2)
            
            radio = tk.Radiobutton(frame, text=classifier, variable=self.classifier_var, 
                                 value=classifier, command=self.on_classifier_change,
                                 font=('Arial', 10, 'bold'), bg='#f0f0f0',
                                 activebackground='#e8f4fd')
            radio.pack(anchor='w')
            
            desc_label = tk.Label(frame, text=description, font=('Arial', 8), 
                                bg='#f0f0f0', fg='#555', justify='left')
            desc_label.pack(anchor='w', padx=(20, 0))
        
        # System status
        status_frame = tk.LabelFrame(left_frame, text="System Status", 
                                   font=('Arial', 10, 'bold'), bg='#f0f0f0')
        status_frame.pack(fill='x', pady=(0, 15))
        
        self.status_label = tk.Label(status_frame, text="Initializing...", 
                                   font=('Arial', 9), bg='#f0f0f0', fg='#2c3e50')
        self.status_label.pack(anchor='w', pady=2)
        
        self.enrolled_label = tk.Label(status_frame, text="Enrolled: 0 people", 
                                     font=('Arial', 9), bg='#f0f0f0', fg='#2c3e50')
        self.enrolled_label.pack(anchor='w', pady=2)
        
        self.confidence_label = tk.Label(status_frame, text="Confidence: 0.7", 
                                       font=('Arial', 9), bg='#f0f0f0', fg='#2c3e50')
        self.confidence_label.pack(anchor='w', pady=2)
        
        # Control buttons
        button_frame = tk.LabelFrame(left_frame, text="Actions", 
                                   font=('Arial', 10, 'bold'), bg='#f0f0f0')
        button_frame.pack(fill='x', pady=(0, 15))
        
        # Start/Stop camera button
        self.camera_button = tk.Button(button_frame, text="▶ Start Camera", 
                                     command=self.toggle_camera,
                                     font=('Arial', 11, 'bold'), bg='#27ae60', fg='white',
                                     activebackground='#2ecc71', cursor='hand2')
        self.camera_button.pack(fill='x', pady=5)
        
        # Enroll button
        enroll_button = tk.Button(button_frame, text="👤 Enroll Students", 
                                command=self.enroll_students,
                                font=('Arial', 11, 'bold'), bg='#3498db', fg='white',
                                activebackground='#5dade2', cursor='hand2')
        enroll_button.pack(fill='x', pady=5)
        
        # Generate report button
        report_button = tk.Button(button_frame, text="📊 Generate Report", 
                                command=self.generate_report,
                                font=('Arial', 11, 'bold'), bg='#f39c12', fg='white',
                                activebackground='#f4b942', cursor='hand2')
        report_button.pack(fill='x', pady=5)
        
        # Compare classifiers button
        compare_button = tk.Button(button_frame, text="⚖ Compare Classifiers", 
                                 command=self.compare_classifiers,
                                 font=('Arial', 11, 'bold'), bg='#9b59b6', fg='white',
                                 activebackground='#bb6bd9', cursor='hand2')
        compare_button.pack(fill='x', pady=5)
        
        # Right panel - Logs and status
        right_frame = tk.LabelFrame(main_frame, text="System Logs", 
                                   font=('Arial', 12, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(right_frame, height=25, width=50,
                                                font=('Consolas', 9), bg='#2c3e50', fg='#ecf0f1',
                                                insertbackground='white')
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Clear logs button
        clear_button = tk.Button(right_frame, text="🗑 Clear Logs", 
                               command=self.clear_logs,
                               font=('Arial', 9), bg='#e74c3c', fg='white',
                               activebackground='#ec7063', cursor='hand2')
        clear_button.pack(pady=(0, 10))
    
    def log_message(self, message, level="INFO"):
        """Add message to log queue for thread-safe logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level == "ERROR":
            formatted_msg = f"[{timestamp}] ❌ {message}\n"
        elif level == "SUCCESS":
            formatted_msg = f"[{timestamp}] ✅ {message}\n"
        elif level == "WARNING":
            formatted_msg = f"[{timestamp}] ⚠️ {message}\n"
        else:
            formatted_msg = f"[{timestamp}] ℹ️ {message}\n"
        
        self.log_queue.put(formatted_msg)
    
    def process_log_queue(self):
        """Process log messages from queue"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_log_queue)
    
    def clear_logs(self):
        """Clear the log display"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Logs cleared", "INFO")
    
    def create_classifier(self, classifier_type):
        """Create a classifier instance based on type"""
        if classifier_type == 'SVM':
            return SVMClassifier(kernel='linear', C=1.0, similarity_weight=0.5)
        elif classifier_type == 'KNN':
            return KNNClassifier(n_neighbors=5, weights='distance', 
                               metric='euclidean', similarity_weight=0.5)
        elif classifier_type == 'LogisticRegression':
            return LogisticRegressionClassifier(C=1.0, similarity_weight=0.5)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def initialize_system(self):
        """Initialize the face recognition system"""
        try:
            self.log_message("Initializing Face Recognition System...", "INFO")
            
            classifier = self.create_classifier(self.current_classifier)
            self.system = FaceRecognitionSystem(
                classifier=classifier,
                model_path='models/',
                database_path='attendance.db',
                confidence_threshold=0.7
            )
            
            # Update status
            enrolled_count = len(set(self.system.known_names)) if self.system.known_names else 0
            self.status_label.config(text=f"✅ Ready ({self.current_classifier})")
            self.enrolled_label.config(text=f"Enrolled: {enrolled_count} people")
            
            self.log_message(f"System initialized with {self.current_classifier}", "SUCCESS")
            self.log_message(f"Enrolled users: {enrolled_count}", "INFO")
            
        except Exception as e:
            self.log_message(f"Failed to initialize system: {e}", "ERROR")
            self.status_label.config(text="❌ Initialization Failed")
    
    def on_classifier_change(self):
        """Handle classifier selection change"""
        new_classifier = self.classifier_var.get()
        if new_classifier != self.current_classifier:
            self.log_message(f"Switching from {self.current_classifier} to {new_classifier}...", "INFO")
            
            if self.camera_running:
                self.toggle_camera()  # Stop camera first
            
            old_classifier = self.current_classifier
            self.current_classifier = new_classifier
            
            try:
                # Create new classifier
                classifier = self.create_classifier(new_classifier)
                
                # Switch classifier in system
                if self.system and self.system.switch_classifier(classifier):
                    self.log_message(f"Successfully switched to {new_classifier}", "SUCCESS")
                    self.status_label.config(text=f"✅ Ready ({new_classifier})")
                else:
                    self.log_message(f"Failed to switch to {new_classifier}", "ERROR")
                    # Revert selection
                    self.classifier_var.set(old_classifier)
                    self.current_classifier = old_classifier
                    
            except Exception as e:
                self.log_message(f"Error switching classifier: {e}", "ERROR")
                # Revert selection
                self.classifier_var.set(old_classifier)
                self.current_classifier = old_classifier
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera and face recognition"""
        if not self.system:
            self.log_message("System not initialized", "ERROR")
            return
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.log_message("Failed to open camera", "ERROR")
                return
            
            self.camera_running = True
            self.camera_button.config(text="⏹ Stop Camera", bg='#e74c3c', activebackground='#ec7063')
            self.log_message("Camera started - Face recognition active", "SUCCESS")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            self.log_message(f"Failed to start camera: {e}", "ERROR")
    
    def stop_camera(self):
        """Stop the camera"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        self.camera_button.config(text="▶ Start Camera", bg='#27ae60', activebackground='#2ecc71')
        self.log_message("Camera stopped", "INFO")
    
    def camera_loop(self):
        """Main camera processing loop"""
        frame_count = 0
        
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process every 5th frame for performance
            frame_count += 1
            if frame_count % 5 != 0:
                cv2.imshow('Face Recognition Attendance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Create display frame
            display_frame = frame.copy()
            
            # Detect faces
            boxes = self.system.extractor.detect_faces(frame)
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Extract face for recognition
                    face_crop = frame[max(0, y1):min(frame.shape[0], y2), 
                                    max(0, x1):min(frame.shape[1], x2)]
                    
                    if face_crop.size > 0:
                        # Get embedding and predict
                        embedding = self.system.extractor.extract_face_embedding(face_crop)
                        
                        if embedding is not None:
                            predicted_name, confidence = self.system.predict_person(embedding)
                            
                            # Choose color based on recognition
                            if predicted_name != "Unknown":
                                color = (0, 255, 0)  # Green for recognized
                                
                                # Try to mark attendance
                                if confidence >= self.system.confidence_threshold:
                                    message, marked = self.system.mark_attendance(predicted_name, confidence)
                                    if marked:
                                        self.log_message(f"Attendance marked: {predicted_name} ({confidence:.3f})", "SUCCESS")
                                else:
                                    self.log_message(f"Low confidence: {predicted_name} ({confidence:.3f})", "WARNING")
                            else:
                                color = (0, 0, 255)  # Red for unknown
                            
                            # Draw rectangle and text
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw name and confidence
                            label = f"{predicted_name} ({confidence:.2f})"
                            cv2.putText(display_frame, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add system info
            unique_people = len(set(self.system.known_names)) if self.system.known_names else 0
            cv2.putText(display_frame, f"Classifier: {self.current_classifier} | Enrolled: {unique_people}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Face Recognition Attendance', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop_camera()
    
    def enroll_students(self):
        """Enroll students from image folder"""
        try:
            self.log_message("Starting batch enrollment from imageFolder...", "INFO")
            
            # Use threading to prevent GUI freeze
            def enroll_thread():
                self.batch_enroll_from_folder("imageFolder")
            
            threading.Thread(target=enroll_thread, daemon=True).start()
            
        except Exception as e:
            self.log_message(f"Enrollment failed: {e}", "ERROR")
    
    def batch_enroll_from_folder(self, batch_folder):
        """Enroll multiple people from folder structure"""
        batch_path = Path(batch_folder)
        if not batch_path.exists():
            self.log_message(f"Batch folder not found: {batch_folder}", "ERROR")
            return
        
        self.log_message(f"Scanning for people in: {batch_folder}", "INFO")
        
        # Find all subdirectories
        person_folders = [f for f in batch_path.iterdir() if f.is_dir()]
        
        if not person_folders:
            self.log_message("No person folders found!", "ERROR")
            return
        
        self.log_message(f"Found {len(person_folders)} people to enroll", "INFO")
        
        # Enroll each person
        successful_enrollments = 0
        
        for person_folder in person_folders:
            person_name = person_folder.name
            self.log_message(f"Enrolling {person_name}...", "INFO")
            
            # Count images in folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [f for f in person_folder.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            if not image_files:
                self.log_message(f"No images found for {person_name}", "WARNING")
                continue
            
            # Enroll person
            success = self.system.enroll_person(
                name=person_name,
                images_path=str(person_folder),
                webcam_capture=False,
                num_samples=0
            )
            
            if success:
                self.log_message(f"Successfully enrolled {person_name} ({len(image_files)} images)", "SUCCESS")
                successful_enrollments += 1
            else:
                self.log_message(f"Failed to enroll {person_name}", "ERROR")
        
        self.log_message(f"Batch enrollment complete: {successful_enrollments}/{len(person_folders)} people", "SUCCESS")
        
        # Update status
        enrolled_count = len(set(self.system.known_names)) if self.system.known_names else 0
        self.enrolled_label.config(text=f"Enrolled: {enrolled_count} people")
    
    def generate_report(self):
        """Generate attendance report"""
        try:
            self.log_message("Generating attendance report...", "INFO")
            
            df = self.system.generate_report()
            
            if not df.empty:
                # Show statistics
                total_records = len(df)
                unique_people = df['name'].nunique()
                avg_confidence = df['confidence'].mean()
                
                self.log_message(f"Report generated successfully", "SUCCESS")
                self.log_message(f"Total records: {total_records}", "INFO")
                self.log_message(f"Unique people: {unique_people}", "INFO")
                self.log_message(f"Average confidence: {avg_confidence:.3f}", "INFO")
                
                # Show classifier usage if available
                if 'classifier_type' in df.columns:
                    classifier_usage = df['classifier_type'].value_counts()
                    self.log_message("Classifier usage:", "INFO")
                    for classifier, count in classifier_usage.items():
                        self.log_message(f"  {classifier}: {count} records", "INFO")
                
                # Save to CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"attendance_report_{timestamp}.csv"
                df.to_csv(filename, index=False)
                self.log_message(f"Report saved to: {filename}", "SUCCESS")
                
            else:
                self.log_message("No attendance records found", "WARNING")
                
        except Exception as e:
            self.log_message(f"Failed to generate report: {e}", "ERROR")
    
    def compare_classifiers(self):
        """Compare different classifiers"""
        try:
            self.log_message("Starting classifier comparison...", "INFO")
            
            if not self.system.known_embeddings:
                self.log_message("No enrolled data found for comparison", "ERROR")
                return
            
            # Test different classifiers
            classifiers_to_test = [
                ('SVM', SVMClassifier(kernel='linear', C=1.0, similarity_weight=0.5)),
                ('KNN', KNNClassifier(n_neighbors=5, weights='distance', metric='euclidean', similarity_weight=0.5)),
                ('LogisticRegression', LogisticRegressionClassifier(C=1.0, similarity_weight=0.5))
            ]
            
            original_classifier = self.system.classifier
            
            for classifier_name, classifier in classifiers_to_test:
                self.log_message(f"Testing {classifier_name}...", "INFO")
                
                # Switch to this classifier
                success = self.system.switch_classifier(classifier)
                if success:
                    params = classifier.get_parameters()
                    self.log_message(f"✅ {classifier_name}: {params.get('num_classes', 0)} classes, {params.get('total_samples', 0)} samples", "SUCCESS")
                else:
                    self.log_message(f"❌ {classifier_name}: Training failed", "ERROR")
            
            # Revert to original classifier
            self.system.switch_classifier(original_classifier)
            self.log_message("Classifier comparison complete", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"Comparison failed: {e}", "ERROR")
    
    def on_closing(self):
        """Handle window closing"""
        if self.camera_running:
            self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
