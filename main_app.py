# gui_main_app.py - Clean & Organized Multi-Classifier Face Recognition GUI
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import queue
import cv2
from PIL import Image, ImageTk
import pandas as pd
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import time

# Add classifiers directory to path
sys.path.append('classifiers')

from face_core import FaceRecognitionSystem
from svm_classifier import SVMClassifier
from knn_classifier import KNNClassifier
from logistic_regression import LogisticRegressionClassifier

class CleanFaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.init_variables()
        self.init_system()
        self.create_gui()
        self.start_message_processing()
    
    def setup_window(self):
        """Setup main window properties"""
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
    
    def init_variables(self):
        """Initialize all GUI variables"""
        # System variables
        self.system = None
        self.video_capture = None
        self.attendance_running = False
        
        # Classifier variables
        self.current_classifier_type = "svm"
        self.available_classifiers = {
            "svm": {"name": "SVM Classifier", "class": SVMClassifier},
            "knn": {"name": "KNN Classifier", "class": KNNClassifier},
            "logistic": {"name": "Logistic Regression", "class": LogisticRegressionClassifier}
        }
        
        # GUI variables
        self.message_queue = queue.Queue()
        self.current_frame = None
        
        # Tkinter variables
        self.classifier_var = tk.StringVar(value="svm")
        self.confidence_threshold_var = tk.DoubleVar(value=0.7)
        self.batch_folder_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.status_var = tk.StringVar(value="System Ready")
    
    def init_system(self, classifier_type="svm"):
        """Initialize face recognition system with specified classifier"""
        try:
            classifier_info = self.available_classifiers[classifier_type]
            classifier = classifier_info["class"]()
            
            self.system = FaceRecognitionSystem(
                classifier=classifier,
                model_path='models/',
                database_path='attendance.db',
                confidence_threshold=self.confidence_threshold_var.get()
            )
            
            self.current_classifier_type = classifier_type
            self.status_var.set(f"Ready - {classifier_info['name']}")
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize system: {e}")
    
    def create_gui(self):
        """Create the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Header
        self.create_header(main_frame)
        
        # Main content (tabs)
        self.create_tabs(main_frame)
        
        # Footer
        self.create_footer(main_frame)
    
    def create_header(self, parent):
        """Create clean header with system info and controls"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', pady=(0, 15))
        
        # Left side - Title and system info
        left_frame = ttk.Frame(header_frame)
        left_frame.pack(side='left', fill='x', expand=True)
        
        # Title
        title_label = ttk.Label(left_frame, text="Face Recognition Attendance System", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(anchor='w')
        
        # System info
        self.system_info_label = ttk.Label(left_frame, text="Loading...", 
                                          foreground='gray')
        self.system_info_label.pack(anchor='w')
        
        # Right side - Classifier selection
        right_frame = ttk.LabelFrame(header_frame, text="Active Classifier", padding="10")
        right_frame.pack(side='right')
        
        # Classifier dropdown and switch button
        classifier_frame = ttk.Frame(right_frame)
        classifier_frame.pack()
        
        classifier_combo = ttk.Combobox(classifier_frame, textvariable=self.classifier_var,
                                       values=list(self.available_classifiers.keys()),
                                       state="readonly", width=12)
        classifier_combo.pack(side='left', padx=(0, 10))
        
        switch_btn = ttk.Button(classifier_frame, text="Switch", 
                               command=self.switch_classifier)
        switch_btn.pack(side='left')
        
        # Confidence threshold
        conf_frame = ttk.Frame(right_frame)
        conf_frame.pack(pady=(10, 0))
        
        ttk.Label(conf_frame, text="Confidence:").pack(side='left')
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, 
                              variable=self.confidence_threshold_var,
                              orient='horizontal', length=100)
        conf_scale.pack(side='left', padx=(5, 5))
        
        self.conf_label = ttk.Label(conf_frame, text="0.70")
        self.conf_label.pack(side='left')
        conf_scale.configure(command=self.update_confidence_label)
        
        self.update_system_info()
    
    def create_tabs(self, parent):
        """Create organized tab interface"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True, pady=(10, 0))
        
        # Create tabs in logical order
        self.create_enrollment_tab()
        self.create_attendance_tab()
        self.create_analysis_tab()
        self.create_reports_tab()
    
    def create_enrollment_tab(self):
        """Create clean enrollment interface"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="👥 Enrollment")
        
        # Main container with padding
        main_container = ttk.Frame(tab_frame, padding="20")
        main_container.pack(fill='both', expand=True)
        
        # Section 1: Batch Enrollment
        batch_section = ttk.LabelFrame(main_container, text="Batch Enrollment", padding="15")
        batch_section.pack(fill='x', pady=(0, 15))
        
        ttk.Label(batch_section, text="Select folder containing person subfolders with images:",
                 foreground='gray').pack(anchor='w')
        
        # Folder selection
        folder_frame = ttk.Frame(batch_section)
        folder_frame.pack(fill='x', pady=(10, 10))
        
        folder_entry = ttk.Entry(folder_frame, textvariable=self.batch_folder_var)
        folder_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        browse_btn = ttk.Button(folder_frame, text="Browse", 
                               command=self.browse_batch_folder)
        browse_btn.pack(side='right', padx=(0, 10))
        
        start_batch_btn = ttk.Button(folder_frame, text="Start Batch Enrollment",
                                    style='Accent.TButton',
                                    command=self.start_batch_enrollment)
        start_batch_btn.pack(side='right')
        
        # Section 2: Individual Enrollment
        individual_section = ttk.LabelFrame(main_container, text="Individual Enrollment", padding="15")
        individual_section.pack(fill='x', pady=(0, 15))
        
        # Name input
        name_frame = ttk.Frame(individual_section)
        name_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(name_frame, text="Person Name:").pack(side='left')
        name_entry = ttk.Entry(name_frame, textvariable=self.name_var, width=30)
        name_entry.pack(side='left', padx=(10, 0))
        
        # Enrollment buttons
        btn_frame = ttk.Frame(individual_section)
        btn_frame.pack(fill='x')
        
        webcam_btn = ttk.Button(btn_frame, text="📷 Webcam Enrollment",
                               command=self.start_webcam_enrollment)
        webcam_btn.pack(side='left', padx=(0, 10))
        
        images_btn = ttk.Button(btn_frame, text="📁 From Images",
                               command=self.enroll_from_images)
        images_btn.pack(side='left')
        
        # Section 3: Progress and Results
        results_section = ttk.LabelFrame(main_container, text="Enrollment Status", padding="15")
        results_section.pack(fill='both', expand=True)
        
        # Progress bar
        self.enrollment_progress = ttk.Progressbar(results_section, mode='indeterminate')
        self.enrollment_progress.pack(fill='x', pady=(0, 10))
        
        # Results text
        self.enrollment_results = scrolledtext.ScrolledText(results_section, height=8, 
                                                           wrap=tk.WORD, font=('Consolas', 9))
        self.enrollment_results.pack(fill='both', expand=True)
    
    def create_attendance_tab(self):
        """Create clean attendance interface"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="📹 Live Attendance")
        
        main_container = ttk.Frame(tab_frame, padding="20")
        main_container.pack(fill='both', expand=True)
        
        # Controls section
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(fill='x', pady=(0, 15))
        
        # Left side - status
        status_frame = ttk.Frame(controls_frame)
        status_frame.pack(side='left')
        
        self.attendance_status_label = ttk.Label(status_frame, text="Ready to start attendance",
                                                font=('Arial', 12))
        self.attendance_status_label.pack()
        
        # Right side - control buttons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(side='right')
        
        self.start_btn = ttk.Button(buttons_frame, text="▶ Start Attendance",
                                   style='Accent.TButton',
                                   command=self.start_attendance)
        self.start_btn.pack(side='left', padx=(0, 10))
        
        self.stop_btn = ttk.Button(buttons_frame, text="⏹ Stop",
                                  command=self.stop_attendance, state='disabled')
        self.stop_btn.pack(side='left')
        
        # Video section
        video_section = ttk.LabelFrame(main_container, text="Camera Feed", padding="10")
        video_section.pack(fill='both', expand=True, pady=(0, 15))
        
        self.video_label = ttk.Label(video_section, text="Camera feed will appear here",
                                    background='black', foreground='white',
                                    font=('Arial', 14), anchor='center')
        self.video_label.pack(fill='both', expand=True)
        
        # Today's attendance section
        attendance_section = ttk.LabelFrame(main_container, text="Today's Attendance", padding="10")
        attendance_section.pack(fill='x')
        
        # Attendance table
        columns = ('Time', 'Name', 'Confidence', 'Classifier')
        self.attendance_tree = ttk.Treeview(attendance_section, columns=columns, 
                                           show='headings', height=5)
        
        # Configure columns
        self.attendance_tree.column('Time', width=80)
        self.attendance_tree.column('Name', width=120)
        self.attendance_tree.column('Confidence', width=100)
        self.attendance_tree.column('Classifier', width=100)
        
        for col in columns:
            self.attendance_tree.heading(col, text=col)
        
        # Scrollbar for attendance table
        attendance_scroll = ttk.Scrollbar(attendance_section, orient='vertical',
                                         command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=attendance_scroll.set)
        
        self.attendance_tree.pack(side='left', fill='both', expand=True)
        attendance_scroll.pack(side='right', fill='y')
        
        self.refresh_attendance_log()
    
    def create_analysis_tab(self):
        """Create classifier analysis and comparison interface"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="📊 Analysis")
        
        main_container = ttk.Frame(tab_frame, padding="20")
        main_container.pack(fill='both', expand=True)
        
        # Controls section
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(controls_frame, text="Performance Analysis & Comparison",
                 font=('Arial', 14, 'bold')).pack(side='left')
        
        # Analysis buttons
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(side='right')
        
        compare_btn = ttk.Button(btn_frame, text="🔍 Compare All Classifiers",
                                style='Accent.TButton',
                                command=self.compare_classifiers)
        compare_btn.pack(side='left', padx=(0, 10))
        
        analysis_btn = ttk.Button(btn_frame, text="📈 Current Classifier Analysis",
                                 command=self.analyze_current_classifier)
        analysis_btn.pack(side='left', padx=(0, 10))
        
        export_btn = ttk.Button(btn_frame, text="💾 Export Results",
                               command=self.export_analysis)
        export_btn.pack(side='left')
        
        # Results section
        results_notebook = ttk.Notebook(main_container)
        results_notebook.pack(fill='both', expand=True)
        
        # Comparison results tab
        comparison_frame = ttk.Frame(results_notebook)
        results_notebook.add(comparison_frame, text="Comparison Results")
        
        # Comparison table
        comp_columns = ('Classifier', 'Accuracy', 'Avg Confidence', 'Training Time', 'Speed')
        self.comparison_tree = ttk.Treeview(comparison_frame, columns=comp_columns,
                                           show='headings', height=8)
        
        for col in comp_columns:
            self.comparison_tree.heading(col, text=col)
            self.comparison_tree.column(col, width=120)
        
        comp_scroll = ttk.Scrollbar(comparison_frame, orient='vertical',
                                   command=self.comparison_tree.yview)
        self.comparison_tree.configure(yscrollcommand=comp_scroll.set)
        
        self.comparison_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        comp_scroll.pack(side='right', fill='y', pady=10)
        
        # Detailed analysis tab
        details_frame = ttk.Frame(results_notebook)
        results_notebook.add(details_frame, text="Detailed Analysis")
        
        self.analysis_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD,
                                                      font=('Consolas', 9))
        self.analysis_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_reports_tab(self):
        """Create clean reports interface"""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="📋 Reports")
        
        main_container = ttk.Frame(tab_frame, padding="20")
        main_container.pack(fill='both', expand=True)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(header_frame, text="Attendance Reports & Data",
                 font=('Arial', 14, 'bold')).pack(side='left')
        
        # Control buttons
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side='right')
        
        refresh_btn = ttk.Button(btn_frame, text="🔄 Refresh",
                                command=self.refresh_reports)
        refresh_btn.pack(side='left', padx=(0, 10))
        
        export_btn = ttk.Button(btn_frame, text="📄 Export CSV",
                               command=self.export_reports)
        export_btn.pack(side='left')
        
        # Reports content
        reports_notebook = ttk.Notebook(main_container)
        reports_notebook.pack(fill='both', expand=True)
        
        # Attendance records
        attendance_frame = ttk.Frame(reports_notebook)
        reports_notebook.add(attendance_frame, text="Attendance Records")
        
        att_columns = ('ID', 'Name', 'Date', 'Time', 'Confidence', 'Classifier')
        self.reports_tree = ttk.Treeview(attendance_frame, columns=att_columns,
                                        show='headings')
        
        for col in att_columns:
            self.reports_tree.heading(col, text=col)
            self.reports_tree.column(col, width=100)
        
        reports_scroll = ttk.Scrollbar(attendance_frame, orient='vertical',
                                      command=self.reports_tree.yview)
        self.reports_tree.configure(yscrollcommand=reports_scroll.set)
        
        self.reports_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        reports_scroll.pack(side='right', fill='y', pady=10)
        
        # Enrolled users
        users_frame = ttk.Frame(reports_notebook)
        reports_notebook.add(users_frame, text="Enrolled Users")
        
        users_columns = ('ID', 'Name', 'Enrollment Date', 'Samples')
        self.users_tree = ttk.Treeview(users_frame, columns=users_columns,
                                      show='headings')
        
        for col in users_columns:
            self.users_tree.heading(col, text=col)
            self.users_tree.column(col, width=150)
        
        users_scroll = ttk.Scrollbar(users_frame, orient='vertical',
                                    command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=users_scroll.set)
        
        self.users_tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        users_scroll.pack(side='right', fill='y', pady=10)
        
        # Load initial data
        self.refresh_reports()
    
    def create_footer(self, parent):
        """Create clean footer with status"""
        footer_frame = ttk.Frame(parent)
        footer_frame.pack(fill='x', pady=(10, 0))
        
        # Status bar
        status_bar = ttk.Label(footer_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W,
                              font=('Arial', 9))
        status_bar.pack(fill='x')
    
    # Event Handlers and Helper Methods
    def update_confidence_label(self, value):
        """Update confidence threshold label and system"""
        conf_val = float(value)
        self.conf_label.config(text=f"{conf_val:.2f}")
        if self.system:
            self.system.confidence_threshold = conf_val
    
    def update_system_info(self):
        """Update system information display"""
        if self.system:
            enrolled_count = len(set(self.system.known_names)) if self.system.known_names else 0
            classifier_name = self.available_classifiers[self.current_classifier_type]["name"]
            self.system_info_label.config(
                text=f"{classifier_name} • {enrolled_count} users enrolled • Threshold: {self.confidence_threshold_var.get():.2f}"
            )
    
    def switch_classifier(self):
        """Switch to selected classifier"""
        new_type = self.classifier_var.get()
        
        if new_type == self.current_classifier_type:
            messagebox.showinfo("Info", "Already using this classifier")
            return
        
        if not self.system or not self.system.known_embeddings:
            messagebox.showwarning("Warning", "No enrolled users found. Please enroll people first.")
            return
        
        # Confirm switch
        old_name = self.available_classifiers[self.current_classifier_type]["name"]
        new_name = self.available_classifiers[new_type]["name"]
        
        result = messagebox.askyesno("Switch Classifier",
                                   f"Switch from {old_name} to {new_name}?\n\n"
                                   f"This will retrain using the same enrolled data.")
        
        if result:
            try:
                self.status_var.set("Switching classifier...")
                
                # Store current data
                old_embeddings = self.system.known_embeddings.copy()
                old_names = self.system.known_names.copy()
                
                # Initialize new classifier
                self.init_system(new_type)
                
                # Transfer data and retrain
                self.system.known_embeddings = old_embeddings
                self.system.known_names = old_names
                
                success = self.system.train_classifier()
                if success:
                    self.system.save_system()
                    self.update_system_info()
                    self.status_var.set(f"Switched to {new_name}")
                    messagebox.showinfo("Success", f"Successfully switched to {new_name}")
                else:
                    raise Exception("Failed to train new classifier")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to switch classifier: {e}")
                self.classifier_var.set(self.current_classifier_type)
                self.status_var.set("Classifier switch failed")
    
    # Enrollment Methods
    def browse_batch_folder(self):
        """Browse for batch enrollment folder"""
        folder = filedialog.askdirectory(title="Select folder containing person subfolders")
        if folder:
            self.batch_folder_var.set(folder)
    
    def start_batch_enrollment(self):
        """Start batch enrollment process"""
        folder = self.batch_folder_var.get()
        if not folder:
            messagebox.showwarning("Warning", "Please select a folder first")
            return
        
        if not os.path.exists(folder):
            messagebox.showerror("Error", "Selected folder does not exist")
            return
        
        # Clear results and start progress
        self.enrollment_results.delete(1.0, tk.END)
        self.enrollment_progress.start()
        
        # Start in thread
        thread = threading.Thread(target=self.batch_enrollment_worker, args=(folder,))
        thread.daemon = True
        thread.start()
    
    def batch_enrollment_worker(self, folder):
        """Worker thread for batch enrollment"""
        try:
            self.message_queue.put(("status", "Starting batch enrollment..."))
            self.message_queue.put(("result", f"📁 Scanning folder: {folder}\n"))
            
            folder_path = Path(folder)
            person_folders = [f for f in folder_path.iterdir() if f.is_dir()]
            
            if not person_folders:
                self.message_queue.put(("result", "❌ No person folders found!\n"))
                return
            
            self.message_queue.put(("result", f"👥 Found {len(person_folders)} people to enroll:\n"))
            for folder in person_folders:
                self.message_queue.put(("result", f"   • {folder.name}\n"))
            
            self.message_queue.put(("result", "\n🚀 Starting enrollment process...\n\n"))
            
            successful = 0
            for person_folder in person_folders:
                person_name = person_folder.name
                self.message_queue.put(("result", f"📝 Enrolling {person_name}... "))
                
                success = self.system.enroll_person(
                    name=person_name,
                    images_path=str(person_folder),
                    webcam_capture=False,
                    num_samples=0
                )
                
                if success:
                    successful += 1
                    self.message_queue.put(("result", "✅ Success\n"))
                else:
                    self.message_queue.put(("result", "❌ Failed\n"))
            
            self.message_queue.put(("result", f"\n🎉 Batch enrollment complete!\n"))
            self.message_queue.put(("result", f"📊 Successfully enrolled: {successful}/{len(person_folders)} people\n"))
            self.message_queue.put(("status", f"Batch enrollment complete - {successful} people enrolled"))
            
        except Exception as e:
            self.message_queue.put(("result", f"❌ Error during batch enrollment: {e}\n"))
            self.message_queue.put(("status", "Batch enrollment failed"))
        finally:
            self.message_queue.put(("progress_stop", None))
            self.message_queue.put(("update_info", None))
    
    def start_webcam_enrollment(self):
        """Start webcam enrollment"""
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name first")
            return
        
        self.enrollment_results.delete(1.0, tk.END)
        self.message_queue.put(("result", f"📷 Starting webcam enrollment for {name}...\n"))
        
        thread = threading.Thread(target=self.webcam_enrollment_worker, args=(name,))
        thread.daemon = True
        thread.start()
    
    def webcam_enrollment_worker(self, name):
        """Worker for webcam enrollment"""
        try:
            success = self.system.enroll_person(
                name=name,
                webcam_capture=True,
                num_samples=10
            )
            
            if success:
                self.message_queue.put(("result", f"✅ Successfully enrolled {name}\n"))
            else:
                self.message_queue.put(("result", f"❌ Failed to enroll {name}\n"))
                
        except Exception as e:
            self.message_queue.put(("result", f"❌ Error enrolling {name}: {e}\n"))
        finally:
            self.message_queue.put(("update_info", None))
    
    def enroll_from_images(self):
        """Enroll from image folder"""
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name first")
            return
        
        folder = filedialog.askdirectory(title=f"Select image folder for {name}")
        if not folder:
            return
        
        self.enrollment_results.delete(1.0, tk.END)
        thread = threading.Thread(target=self.image_enrollment_worker, args=(name, folder))
        thread.daemon = True
        thread.start()
    
    def image_enrollment_worker(self, name, folder):
        """Worker for image enrollment"""
        try:
            self.message_queue.put(("result", f"📁 Enrolling {name} from {folder}...\n"))
            
            success = self.system.enroll_person(
                name=name,
                images_path=folder,
                webcam_capture=False,
                num_samples=0
            )
            
            if success:
                self.message_queue.put(("result", f"✅ Successfully enrolled {name}\n"))
            else:
                self.message_queue.put(("result", f"❌ Failed to enroll {name}\n"))
                
        except Exception as e:
            self.message_queue.put(("result", f"❌ Error enrolling {name}: {e}\n"))
        finally:
            self.message_queue.put(("update_info", None))
    
    # Attendance Methods
    def start_attendance(self):
        """Start live attendance system"""
        if not self.system or not self.system.known_embeddings:
            messagebox.showwarning("Warning", "No people enrolled. Please enroll people first.")
            return
        
        self.attendance_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.attendance_status_label.config(text="🔴 Live attendance running...", foreground='red')
        
        thread = threading.Thread(target=self.attendance_worker)
        thread.daemon = True
        thread.start()
    
    def stop_attendance(self):
        """Stop live attendance system"""
        self.attendance_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.attendance_status_label.config(text="⏹ Attendance stopped", foreground='gray')
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        self.video_label.config(image='', text="Camera feed stopped")
        self.status_var.set("Attendance system stopped")
    
    def attendance_worker(self):
        """Worker thread for attendance system"""
        self.video_capture = cv2.VideoCapture(0)
        frame_count = 0
        
        try:
            while self.attendance_running:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 5 != 0:  # Process every 5th frame
                    continue
                
                # Process frame for face recognition
                display_frame = frame.copy()
                boxes = self.system.extractor.detect_faces(frame)
                
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Extract face for recognition
                        face_crop = frame[max(0, y1):min(frame.shape[0], y2), 
                                        max(0, x1):min(frame.shape[1], x2)]
                        
                        if face_crop.size > 0:
                            embedding = self.system.extractor.extract_face_embedding(face_crop)
                            
                            if embedding is not None:
                                predicted_name, confidence = self.system.predict_person(embedding)
                                
                                if predicted_name != "Unknown":
                                    color = (0, 255, 0)  # Green
                                    
                                    if confidence >= self.system.confidence_threshold:
                                        message, marked = self.system.mark_attendance(predicted_name, confidence)
                                        if marked:
                                            self.message_queue.put(("attendance", {
                                                'name': predicted_name,
                                                'confidence': confidence,
                                                'time': datetime.now().strftime("%H:%M:%S"),
                                                'classifier': self.system.classifier.get_name()
                                            }))
                                else:
                                    color = (0, 0, 255)  # Red
                                
                                # Draw rectangle and text
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                label = f"{predicted_name} ({confidence:.2f})"
                                cv2.putText(display_frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add system info to display
                info_text = f"Classifier: {self.system.classifier.get_name()} | Threshold: {self.system.confidence_threshold:.2f}"
                cv2.putText(display_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Convert and display frame
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image = pil_image.resize((640, 480))
                photo = ImageTk.PhotoImage(pil_image)
                
                self.message_queue.put(("video_frame", photo))
                
        except Exception as e:
            self.message_queue.put(("status", f"Attendance error: {e}"))
        finally:
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
    
    def refresh_attendance_log(self):
        """Refresh today's attendance log"""
        try:
            # Clear existing items
            for item in self.attendance_tree.get_children():
                self.attendance_tree.delete(item)
            
            # Get today's attendance
            today = datetime.now().date()
            df = self.system.generate_report()
            
            if not df.empty:
                today_df = df[pd.to_datetime(df['date']).dt.date == today]
                
                for _, row in today_df.iterrows():
                    time_str = pd.to_datetime(row['timestamp']).strftime("%H:%M:%S")
                    classifier_type = row.get('classifier_type', 'Unknown')
                    self.attendance_tree.insert('', 0, values=(
                        time_str, row['name'], f"{row['confidence']:.3f}", classifier_type
                    ))
        except Exception as e:
            print(f"Error refreshing attendance log: {e}")
    
    # Analysis Methods
    def compare_classifiers(self):
        """Compare all available classifiers"""
        if not self.system or not self.system.known_embeddings:
            messagebox.showwarning("Warning", "No enrolled users found for comparison")
            return
        
        # Clear previous results
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)
        
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "🔍 Running classifier comparison...\n\n")
        
        # Run comparison in thread
        thread = threading.Thread(target=self.comparison_worker)
        thread.daemon = True
        thread.start()
    
    def comparison_worker(self):
        """Worker thread for classifier comparison"""
        try:
            results = []
            total_samples = len(self.system.known_embeddings)
            unique_people = len(set(self.system.known_names))
            
            self.message_queue.put(("analysis_update", 
                                  f"📊 Dataset: {total_samples} samples, {unique_people} people\n\n"))
            
            # Test each classifier
            for classifier_type, classifier_info in self.available_classifiers.items():
                classifier_name = classifier_info["name"]
                self.message_queue.put(("analysis_update", f"🧠 Testing {classifier_name}...\n"))
                
                try:
                    classifier = classifier_info["class"]()
                    
                    # Measure training time
                    start_time = time.time()
                    success = classifier.train(self.system.known_embeddings, self.system.known_names)
                    training_time = time.time() - start_time
                    
                    if success:
                        # Test accuracy and speed
                        correct = 0
                        confidences = []
                        prediction_times = []
                        
                        for embedding, true_name in zip(self.system.known_embeddings, self.system.known_names):
                            start_pred = time.time()
                            predicted_name, confidence = classifier.predict(embedding)
                            pred_time = time.time() - start_pred
                            
                            prediction_times.append(pred_time)
                            confidences.append(confidence)
                            
                            if predicted_name == true_name:
                                correct += 1
                        
                        accuracy = correct / total_samples
                        avg_confidence = sum(confidences) / len(confidences)
                        avg_pred_time = sum(prediction_times) / len(prediction_times)
                        
                        results.append({
                            'classifier': classifier_name,
                            'type': classifier_type,
                            'accuracy': accuracy,
                            'avg_confidence': avg_confidence,
                            'training_time': training_time,
                            'prediction_time': avg_pred_time
                        })
                        
                        self.message_queue.put(("comparison_result", {
                            'classifier': classifier_name,
                            'accuracy': f"{accuracy:.3f}",
                            'avg_confidence': f"{avg_confidence:.3f}",
                            'training_time': f"{training_time:.2f}s",
                            'speed': f"{avg_pred_time*1000:.1f}ms"
                        }))
                        
                        self.message_queue.put(("analysis_update", f"   ✅ Accuracy: {accuracy:.3f}\n"))
                        
                    else:
                        self.message_queue.put(("analysis_update", f"   ❌ Training failed\n"))
                        
                except Exception as e:
                    self.message_queue.put(("analysis_update", f"   ❌ Error: {e}\n"))
            
            # Generate detailed analysis
            if results:
                results.sort(key=lambda x: x['accuracy'], reverse=True)
                
                analysis = "\n" + "="*50 + "\n"
                analysis += "📈 COMPARISON RESULTS\n"
                analysis += "="*50 + "\n\n"
                
                for i, result in enumerate(results, 1):
                    analysis += f"{i}. {result['classifier']}\n"
                    analysis += f"   🎯 Accuracy: {result['accuracy']:.3f}\n"
                    analysis += f"   🎲 Avg Confidence: {result['avg_confidence']:.3f}\n"
                    analysis += f"   ⏱️ Training Time: {result['training_time']:.2f}s\n"
                    analysis += f"   ⚡ Prediction Speed: {result['prediction_time']*1000:.1f}ms\n\n"
                
                # Recommendations
                best_accuracy = results[0]
                fastest_training = min(results, key=lambda x: x['training_time'])
                fastest_prediction = min(results, key=lambda x: x['prediction_time'])
                
                analysis += "🏆 RECOMMENDATIONS\n"
                analysis += "-" * 30 + "\n"
                analysis += f"🥇 Best Accuracy: {best_accuracy['classifier']} ({best_accuracy['accuracy']:.3f})\n"
                analysis += f"🚀 Fastest Training: {fastest_training['classifier']} ({fastest_training['training_time']:.2f}s)\n"
                analysis += f"⚡ Fastest Prediction: {fastest_prediction['classifier']} ({fastest_prediction['prediction_time']*1000:.1f}ms)\n\n"
                
                # Usage recommendations
                analysis += "💡 USAGE RECOMMENDATIONS\n"
                analysis += "-" * 30 + "\n"
                if best_accuracy['accuracy'] > 0.95:
                    analysis += f"✅ For production use: {best_accuracy['classifier']}\n"
                if fastest_prediction['prediction_time'] < 0.05:
                    analysis += f"⚡ For real-time systems: {fastest_prediction['classifier']}\n"
                if fastest_training['training_time'] < 1.0:
                    analysis += f"🚀 For frequent retraining: {fastest_training['classifier']}\n"
                
                self.message_queue.put(("analysis_details", analysis))
            
        except Exception as e:
            self.message_queue.put(("analysis_update", f"❌ Error during comparison: {e}\n"))
    
    def analyze_current_classifier(self):
        """Analyze current classifier performance"""
        if not self.system or not self.system.known_embeddings:
            messagebox.showwarning("Warning", "No enrolled users found for analysis")
            return
        
        # Create analysis window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title(f"Performance Analysis - {self.system.classifier.get_name()}")
        analysis_window.geometry("700x500")
        
        # Analysis text area
        text_frame = ttk.Frame(analysis_window, padding="20")
        text_frame.pack(fill='both', expand=True)
        
        analysis_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=('Consolas', 9))
        analysis_text.pack(fill='both', expand=True)
        
        # Run analysis in thread
        thread = threading.Thread(target=self.current_analysis_worker, args=(analysis_text,))
        thread.daemon = True
        thread.start()
    
    def current_analysis_worker(self, text_widget):
        """Worker for current classifier analysis"""
        try:
            classifier_name = self.system.classifier.get_name()
            text_widget.insert(tk.END, f"🔍 Analyzing {classifier_name}...\n\n")
            text_widget.update()
            
            # Basic metrics
            total_samples = len(self.system.known_embeddings)
            unique_people = len(set(self.system.known_names))
            
            text_widget.insert(tk.END, "📊 SYSTEM OVERVIEW\n")
            text_widget.insert(tk.END, "-" * 30 + "\n")
            text_widget.insert(tk.END, f"Classifier: {classifier_name}\n")
            text_widget.insert(tk.END, f"Enrolled People: {unique_people}\n")
            text_widget.insert(tk.END, f"Total Samples: {total_samples}\n")
            text_widget.insert(tk.END, f"Confidence Threshold: {self.system.confidence_threshold}\n\n")
            
            # Performance test
            correct = 0
            confidences = []
            processing_times = []
            
            for embedding, true_name in zip(self.system.known_embeddings, self.system.known_names):
                start_time = time.time()
                predicted_name, confidence = self.system.predict_person(embedding)
                end_time = time.time()
                
                processing_times.append(end_time - start_time)
                confidences.append(confidence)
                
                if predicted_name == true_name:
                    correct += 1
            
            accuracy = correct / total_samples
            avg_confidence = sum(confidences) / len(confidences)
            avg_processing_time = sum(processing_times) / len(processing_times)
            
            text_widget.insert(tk.END, "🎯 PERFORMANCE RESULTS\n")
            text_widget.insert(tk.END, "-" * 30 + "\n")
            text_widget.insert(tk.END, f"Overall Accuracy: {accuracy:.3f} ({correct}/{total_samples})\n")
            text_widget.insert(tk.END, f"Average Confidence: {avg_confidence:.3f}\n")
            text_widget.insert(tk.END, f"Processing Speed: {avg_processing_time*1000:.1f}ms per face\n")
            text_widget.insert(tk.END, f"Throughput: {1/avg_processing_time:.1f} faces/second\n\n")
            
            # Per-person analysis
            text_widget.insert(tk.END, "👥 PER-PERSON ANALYSIS\n")
            text_widget.insert(tk.END, "-" * 30 + "\n")
            
            person_stats = {}
            for embedding, true_name in zip(self.system.known_embeddings, self.system.known_names):
                if true_name not in person_stats:
                    person_stats[true_name] = {'correct': 0, 'total': 0, 'confidences': []}
                
                predicted_name, confidence = self.system.predict_person(embedding)
                person_stats[true_name]['total'] += 1
                person_stats[true_name]['confidences'].append(confidence)
                
                if predicted_name == true_name:
                    person_stats[true_name]['correct'] += 1
            
            for person, stats in person_stats.items():
                person_accuracy = stats['correct'] / stats['total']
                person_avg_conf = sum(stats['confidences']) / len(stats['confidences'])
                text_widget.insert(tk.END, f"{person}:\n")
                text_widget.insert(tk.END, f"  Accuracy: {person_accuracy:.3f}\n")
                text_widget.insert(tk.END, f"  Avg Confidence: {person_avg_conf:.3f}\n")
                text_widget.insert(tk.END, f"  Samples: {stats['total']}\n\n")
            
            # Recommendations
            text_widget.insert(tk.END, "💡 RECOMMENDATIONS\n")
            text_widget.insert(tk.END, "-" * 30 + "\n")
            
            if accuracy < 0.8:
                text_widget.insert(tk.END, "⚠️ Low accuracy detected. Consider:\n")
                text_widget.insert(tk.END, "   • Adding more training samples\n")
                text_widget.insert(tk.END, "   • Trying a different classifier\n")
                text_widget.insert(tk.END, "   • Adjusting confidence threshold\n\n")
            elif accuracy > 0.95:
                text_widget.insert(tk.END, "✅ Excellent accuracy! This classifier is performing well.\n\n")
            
            if avg_processing_time > 0.1:
                text_widget.insert(tk.END, "⏱️ Consider faster classifier for real-time applications.\n\n")
            
            text_widget.insert(tk.END, "✅ Analysis complete!\n")
            
        except Exception as e:
            text_widget.insert(tk.END, f"❌ Error during analysis: {e}\n")
    
    def export_analysis(self):
        """Export analysis results"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save analysis results"
            )
            
            if filename:
                # Get comparison data
                comparison_data = []
                for item in self.comparison_tree.get_children():
                    values = self.comparison_tree.item(item)['values']
                    comparison_data.append({
                        'classifier': values[0],
                        'accuracy': values[1], 
                        'avg_confidence': values[2],
                        'training_time': values[3],
                        'prediction_speed': values[4]
                    })
                
                # Create export data
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'system_info': {
                        'current_classifier': self.system.classifier.get_name() if self.system else 'None',
                        'dataset_size': len(self.system.known_embeddings) if self.system and self.system.known_embeddings else 0,
                        'unique_people': len(set(self.system.known_names)) if self.system and self.system.known_names else 0,
                        'confidence_threshold': self.confidence_threshold_var.get()
                    },
                    'comparison_results': comparison_data,
                    'detailed_analysis': self.analysis_text.get(1.0, tk.END)
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Analysis results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {e}")
    
    # Reports Methods
    def refresh_reports(self):
        """Refresh all reports"""
        try:
            # Clear existing data
            for item in self.reports_tree.get_children():
                self.reports_tree.delete(item)
            for item in self.users_tree.get_children():
                self.users_tree.delete(item)
            
            if not self.system:
                return
            
            # Refresh attendance records
            df = self.system.generate_report()
            for _, row in df.iterrows():
                date_time = pd.to_datetime(row['timestamp'])
                classifier_type = row.get('classifier_type', 'Unknown')
                self.reports_tree.insert('', 'end', values=(
                    row['id'], row['name'], date_time.strftime("%Y-%m-%d"),
                    date_time.strftime("%H:%M:%S"), f"{row['confidence']:.3f}", classifier_type
                ))
            
            # Refresh enrolled users
            users_df = self.system.database.get_enrolled_users()
            for _, row in users_df.iterrows():
                enrollment_date = pd.to_datetime(row['enrollment_date']).strftime("%Y-%m-%d %H:%M")
                self.users_tree.insert('', 'end', values=(
                    row['id'], row['name'], enrollment_date, row['total_embeddings']
                ))
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh reports: {e}")
    
    def export_reports(self):
        """Export reports to CSV"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save attendance report"
            )
            
            if filename and self.system:
                df = self.system.generate_report()
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Report exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {e}")
    
    # Message Processing
    def start_message_processing(self):
        """Start processing messages from worker threads"""
        self.process_messages()
    
    def process_messages(self):
        """Process messages from worker threads"""
        try:
            while True:
                message_type, data = self.message_queue.get_nowait()
                
                if message_type == "status":
                    self.status_var.set(data)
                
                elif message_type == "result":
                    self.enrollment_results.insert(tk.END, data)
                    self.enrollment_results.see(tk.END)
                
                elif message_type == "progress_stop":
                    self.enrollment_progress.stop()
                
                elif message_type == "update_info":
                    self.update_system_info()
                
                elif message_type == "video_frame":
                    self.video_label.configure(image=data)
                    self.video_label.image = data
                
                elif message_type == "attendance":
                    # Add new attendance record
                    self.attendance_tree.insert('', 0, values=(
                        data['time'], data['name'], f"{data['confidence']:.3f}", data['classifier']
                    ))
                
                elif message_type == "comparison_result":
                    # Add comparison result
                    self.comparison_tree.insert('', 'end', values=(
                        data['classifier'], data['accuracy'], data['avg_confidence'],
                        data['training_time'], data['speed']
                    ))
                
                elif message_type == "analysis_update":
                    self.analysis_text.insert(tk.END, data)
                    self.analysis_text.see(tk.END)
                
                elif message_type == "analysis_details":
                    self.analysis_text.insert(tk.END, data)
                    self.analysis_text.see(tk.END)
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing messages: {e}")
        
        # Schedule next check
        self.root.after(100, self.process_messages)
    
    def on_closing(self):
        """Handle application closing"""
        if self.attendance_running:
            self.stop_attendance()
        
        if self.video_capture:
            self.video_capture.release()
        
        self.root.destroy()

def main():
    """Main function to run the clean GUI application"""
    root = tk.Tk()
    
    # Set modern style
    try:
        # Use Windows 10 style if available
        root.tk.call('source', 'azure.tcl')
        root.tk.call('set_theme', 'light')
    except:
        pass
    
    app = CleanFaceRecognitionGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()