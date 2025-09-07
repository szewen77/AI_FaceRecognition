import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import time
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import json

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from face_core import FixedMultiClassifierSystem
    print("✅ Successfully imported FixedMultiClassifierSystem")
except ImportError as e:
    print(f"❌ Error importing face_core: {e}")
    sys.exit(1)

class FaceRecognitionGUI:
    """Modern GUI for Multi-Classifier Face Recognition System"""
    
    def __init__(self, root):
        self.root = root
        self.system = None
        self.is_initialized = False
        self.attendance_running = False
        
        # Setup main window
        self.setup_main_window()
        
        # Create GUI components
        self.create_widgets()
        
        # Initialize system in background
        self.initialize_system()
    
    def setup_main_window(self):
        """Setup main application window"""
        self.root.title("🎯 Multi-Classifier Face Recognition Attendance System")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'background': '#F5F5F5'
        }
        
        self.root.configure(bg=self.colors['background'])
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_enrollment_tab()
        self.create_attendance_tab()
        self.create_users_tab()
        self.create_reports_tab()
        self.create_settings_tab()
        
        # Status bar
        self.create_status_bar()
        
    def create_dashboard_tab(self):
        """Create dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="📊 Dashboard")
        
        # Title
        title = tk.Label(dashboard_frame, text="Multi-Classifier Face Recognition System", 
                        font=('Arial', 16, 'bold'), bg=self.colors['background'])
        title.pack(pady=10)
        
        # System status frame
        status_frame = ttk.LabelFrame(dashboard_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_labels = {}
        status_info = [
            ("System Status:", "Initializing..."),
            ("Enrolled Users:", "0"),
            ("Total Embeddings:", "0"),
            ("Active Classifiers:", "None"),
            ("Confidence Threshold:", "0.70"),
            ("Verification Threshold:", "0.70")
        ]
        
        for i, (label, value) in enumerate(status_info):
            row = i // 2
            col = (i % 2) * 2
            
            tk.Label(status_frame, text=label, font=('Arial', 10, 'bold')).grid(
                row=row, column=col, sticky='w', padx=10, pady=5)
            
            self.status_labels[label] = tk.Label(status_frame, text=value, font=('Arial', 10))
            self.status_labels[label].grid(row=row, column=col+1, sticky='w', padx=10, pady=5)
        
        # Performance frame
        perf_frame = ttk.LabelFrame(dashboard_frame, text="Classifier Performance", padding=10)
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Performance text widget
        self.performance_text = scrolledtext.ScrolledText(perf_frame, height=8, font=('Courier', 9))
        self.performance_text.pack(fill=tk.BOTH, expand=True)
        self.performance_text.insert(tk.END, "No performance data available yet.\n\nEnroll users and run attendance to see classifier performance.")
        
        # Refresh button
        refresh_btn = ttk.Button(dashboard_frame, text="🔄 Refresh Dashboard", 
                                command=self.refresh_dashboard)
        refresh_btn.pack(pady=10)
        
    def create_enrollment_tab(self):
        """Create enrollment tab"""
        enrollment_frame = ttk.Frame(self.notebook)
        self.notebook.add(enrollment_frame, text="👤 Enrollment")
        
        # Webcam enrollment section
        webcam_frame = ttk.LabelFrame(enrollment_frame, text="Webcam Enrollment", padding=10)
        webcam_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(webcam_frame, text="User Name:").grid(row=0, column=0, sticky='w', pady=5)
        self.webcam_name_var = tk.StringVar()
        tk.Entry(webcam_frame, textvariable=self.webcam_name_var, width=30).grid(row=0, column=1, padx=10, pady=5)
        
        tk.Label(webcam_frame, text="Samples:").grid(row=1, column=0, sticky='w', pady=5)
        self.webcam_samples_var = tk.StringVar(value="10")
        samples_spin = tk.Spinbox(webcam_frame, from_=3, to=50, textvariable=self.webcam_samples_var, width=10)
        samples_spin.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        webcam_btn = ttk.Button(webcam_frame, text="📷 Start Webcam Enrollment", 
                               command=self.start_webcam_enrollment)
        webcam_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Image folder enrollment section
        folder_frame = ttk.LabelFrame(enrollment_frame, text="Image Folder Enrollment", padding=10)
        folder_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(folder_frame, text="User Name:").grid(row=0, column=0, sticky='w', pady=5)
        self.folder_name_var = tk.StringVar()
        tk.Entry(folder_frame, textvariable=self.folder_name_var, width=30).grid(row=0, column=1, padx=10, pady=5)
        
        tk.Label(folder_frame, text="Images Files:").grid(row=1, column=0, sticky='w', pady=5)
        self.files_path_var = tk.StringVar()
        files_path_entry = tk.Entry(folder_frame, textvariable=self.files_path_var, width=40)
        files_path_entry.grid(row=1, column=1, padx=10, pady=5)
        
        browse_btn = ttk.Button(folder_frame, text="📁 Browse", command=self.browse_files)
        browse_btn.grid(row=1, column=2, padx=5, pady=5)
        
        files_btn = ttk.Button(folder_frame, text="📁 Start Folder Enrollment", 
                               command=self.start_file_enrollment)
        files_btn.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Enrollment log
        log_frame = ttk.LabelFrame(enrollment_frame, text="Enrollment Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.enrollment_log = scrolledtext.ScrolledText(log_frame, height=10, font=('Courier', 9))
        self.enrollment_log.pack(fill=tk.BOTH, expand=True)
        
    def create_attendance_tab(self):
        """Create attendance tab"""
        attendance_frame = ttk.Frame(self.notebook)
        self.notebook.add(attendance_frame, text="🎥 Live Attendance")
        
        # Control buttons
        control_frame = ttk.Frame(attendance_frame)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.start_attendance_btn = ttk.Button(control_frame, text="▶️ Start Live Attendance", 
                                              command=self.start_live_attendance)
        self.start_attendance_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_attendance_btn = ttk.Button(control_frame, text="⏹️ Stop Attendance", 
                                             command=self.stop_live_attendance, state=tk.DISABLED)
        self.stop_attendance_btn.pack(side=tk.LEFT, padx=5)
        
        # Attendance status
        status_frame = ttk.LabelFrame(attendance_frame, text="Attendance Status", padding=10)
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.attendance_status_var = tk.StringVar(value="Ready to start attendance monitoring")
        tk.Label(status_frame, textvariable=self.attendance_status_var, font=('Arial', 10)).pack()
        
        # Live attendance log
        log_frame = ttk.LabelFrame(attendance_frame, text="Live Attendance Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.attendance_log = scrolledtext.ScrolledText(log_frame, height=15, font=('Courier', 9))
        self.attendance_log.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = """
User-Friendly Live Attendance Instructions:
• Ensure good lighting conditions
• Position camera at eye level
• Look at the camera - attendance will be marked automatically
• A success popup will appear when attendance is marked
• Camera window stays open for continuous monitoring
• 3-second cooldown prevents duplicate markings

Camera Controls (when live window is active):
• Press 'q' to quit manually
• Clean interface without flickering text
• Multi-classifier system for accurate recognition
        """
        
        instr_frame = ttk.LabelFrame(attendance_frame, text="Instructions", padding=10)
        instr_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(instr_frame, text=instructions.strip(), justify=tk.LEFT, font=('Arial', 9)).pack(anchor='w')
    
    def create_users_tab(self):
        """Create users management tab"""
        users_frame = ttk.Frame(self.notebook)
        self.notebook.add(users_frame, text="👥 Users")
        
        # Control buttons
        control_frame = ttk.Frame(users_frame)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(control_frame, text="🔄 Refresh List", 
                  command=self.refresh_users_list).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="❌ Delete User", 
                  command=self.delete_selected_user).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="📊 View Details", 
                  command=self.view_user_details).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="🧹 Clean Low Samples", 
                  command=self.cleanup_low_samples).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="👥 Add 100 More Students", 
                  command=self.add_more_students).pack(side=tk.LEFT, padx=5)
        
        # Total users count
        self.total_users_label = tk.Label(control_frame, text="Total Users: 0", 
                                        font=('Arial', 10, 'bold'), fg='blue')
        self.total_users_label.pack(side=tk.RIGHT, padx=10)
        
        # Users list
        list_frame = ttk.LabelFrame(users_frame, text="Enrolled Users", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Treeview for users list
        columns = ('Name', 'Samples', 'Enrolled', 'Last Updated')
        self.users_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.users_tree.heading(col, text=col)
            if col == 'Name':
                self.users_tree.column(col, width=200)
            elif col == 'Samples':
                self.users_tree.column(col, width=80)
            else:
                self.users_tree.column(col, width=120)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=scrollbar.set)
        
        self.users_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # User details frame
        details_frame = ttk.LabelFrame(users_frame, text="User Details", padding=10)
        details_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.user_details_text = scrolledtext.ScrolledText(details_frame, height=8, font=('Courier', 9))
        self.user_details_text.pack(fill=tk.BOTH, expand=True)
    
    def create_reports_tab(self):
        """Create reports tab"""
        reports_frame = ttk.Frame(self.notebook)
        self.notebook.add(reports_frame, text="📈 Reports")
        
        # Date range selection
        date_frame = ttk.LabelFrame(reports_frame, text="Date Range", padding=10)
        date_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(date_frame, text="Start Date:").grid(row=0, column=0, sticky='w', pady=5)
        self.start_date_var = tk.StringVar()
        tk.Entry(date_frame, textvariable=self.start_date_var, width=15).grid(row=0, column=1, padx=10, pady=5)
        tk.Label(date_frame, text="(YYYY-MM-DD)").grid(row=0, column=2, sticky='w')
        
        tk.Label(date_frame, text="End Date:").grid(row=1, column=0, sticky='w', pady=5)
        self.end_date_var = tk.StringVar()
        tk.Entry(date_frame, textvariable=self.end_date_var, width=15).grid(row=1, column=1, padx=10, pady=5)
        tk.Label(date_frame, text="(YYYY-MM-DD)").grid(row=1, column=2, sticky='w')
        
        # Report buttons
        btn_frame = ttk.Frame(date_frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="📊 Generate Report", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="💾 Export CSV", 
                  command=self.export_csv).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="🎯 Performance Report", 
                  command=self.show_performance_report).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="📊 Comprehensive Evaluation", 
                  command=self.run_comprehensive_evaluation).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="🔄 Retrain Current Students", 
                  command=self.retrain_and_evaluate).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="📊 Generate Performance Charts", 
                  command=self.generate_performance_charts).pack(side=tk.LEFT, padx=5)
        
        
        # Report display
        report_frame = ttk.LabelFrame(reports_frame, text="Attendance Report", padding=10)
        report_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.report_text = scrolledtext.ScrolledText(report_frame, height=20, font=('Courier', 9))
        self.report_text.pack(fill=tk.BOTH, expand=True)
    
    def create_settings_tab(self):
        """Create settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # Threshold settings
        thresh_frame = ttk.LabelFrame(settings_frame, text="Recognition Thresholds", padding=10)
        thresh_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(thresh_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky='w', pady=5)
        self.confidence_var = tk.DoubleVar(value=0.7)
        confidence_scale = tk.Scale(thresh_frame, from_=0.1, to=1.0, resolution=0.01, 
                                   orient=tk.HORIZONTAL, variable=self.confidence_var, length=200)
        confidence_scale.grid(row=0, column=1, padx=10, pady=5)
        tk.Label(thresh_frame, text="(Higher = More Strict)").grid(row=0, column=2, sticky='w')
        
        tk.Label(thresh_frame, text="Verification Threshold:").grid(row=1, column=0, sticky='w', pady=5)
        self.verification_var = tk.DoubleVar(value=0.7)
        verification_scale = tk.Scale(thresh_frame, from_=0.1, to=1.0, resolution=0.01, 
                                     orient=tk.HORIZONTAL, variable=self.verification_var, length=200)
        verification_scale.grid(row=1, column=1, padx=10, pady=5)
        tk.Label(thresh_frame, text="(Higher = More Strict)").grid(row=1, column=2, sticky='w')
        
        ttk.Button(thresh_frame, text="💾 Apply Settings", 
                  command=self.apply_settings).grid(row=2, column=0, columnspan=3, pady=10)
        
        # System info
        info_frame = ttk.LabelFrame(settings_frame, text="System Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.system_info_text = scrolledtext.ScrolledText(info_frame, height=15, font=('Courier', 9))
        self.system_info_text.pack(fill=tk.BOTH, expand=True)
        
        # System buttons
        system_btn_frame = ttk.Frame(settings_frame)
        system_btn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(system_btn_frame, text="💾 Save System", 
                  command=self.save_system).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(system_btn_frame, text="🔄 Reload System", 
                  command=self.reload_system).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(system_btn_frame, text="📊 Refresh Info", 
                  command=self.refresh_system_info).pack(side=tk.LEFT, padx=5)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Initializing system...")
        status_label = tk.Label(self.status_bar, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Time label
        self.time_var = tk.StringVar()
        time_label = tk.Label(self.status_bar, textvariable=self.time_var, 
                             relief=tk.SUNKEN, anchor=tk.E)
        time_label.pack(side=tk.RIGHT)
        
        self.update_time()
    
    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_var.set(current_time)
        self.root.after(1000, self.update_time)
    
    def initialize_system(self):
        """Initialize the face recognition system"""
        def init_worker():
            try:
                self.system = FixedMultiClassifierSystem(
                    model_path='models/',
                    database_path='attendance.db',
                    confidence_threshold=0.7,
                    verification_threshold=0.7,
                    device=None
                )
                
                self.is_initialized = True
                self.root.after(0, self.on_system_initialized)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_system_error(str(e)))
        
        # Start initialization in background thread
        thread = threading.Thread(target=init_worker, daemon=True)
        thread.start()
    
    def on_system_initialized(self):
        """Called when system is successfully initialized"""
        self.status_var.set("System initialized successfully!")
        self.refresh_dashboard()
        self.refresh_users_list()
        self.refresh_system_info()
        
        # Log to enrollment tab
        self.log_to_enrollment("✅ System initialized successfully!")
        self.log_to_enrollment("📊 Multi-classifier system ready with SVM, KNN, and Logistic Regression")
        
    def on_system_error(self, error_msg):
        """Called when system initialization fails"""
        self.status_var.set("System initialization failed!")
        messagebox.showerror("Initialization Error", f"Failed to initialize system:\n{error_msg}")
        
    def refresh_dashboard(self):
        """Refresh dashboard information"""
        if not self.is_initialized:
            return
        
        try:
            status = self.system.get_system_status()
            
            self.status_labels["System Status:"].config(text="✅ Ready", fg='green')
            self.status_labels["Enrolled Users:"].config(text=str(status['enrolled_users']))
            self.status_labels["Total Embeddings:"].config(text=str(status['total_embeddings']))
            self.status_labels["Active Classifiers:"].config(text=', '.join(status['active_classifiers']))
            self.status_labels["Confidence Threshold:"].config(text=f"{status['confidence_threshold']:.2f}")
            self.status_labels["Verification Threshold:"].config(text=f"{status['verification_threshold']:.2f}")
            
            # Update performance display
            self.performance_text.delete(1.0, tk.END)
            perf_report = self.system.get_classifier_performance_report()
            self.performance_text.insert(tk.END, perf_report)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh dashboard: {e}")
    
    def start_webcam_enrollment(self):
        """Start webcam enrollment"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        name = self.webcam_name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a user name!")
            return
        
        try:
            samples = int(self.webcam_samples_var.get())
        except ValueError:
            samples = 10
        
        def enrollment_worker():
            try:
                self.log_to_enrollment(f"🔄 Starting webcam enrollment for '{name}'...")
                self.log_to_enrollment("📷 Camera window will open - follow on-screen instructions")
                
                success = self.system.enroll_person(
                    name=name,
                    webcam_capture=True,
                    num_samples=samples
                )
                
                if success:
                    self.root.after(0, lambda: self.on_enrollment_success(name))
                else:
                    self.root.after(0, lambda: self.on_enrollment_failed(name))
                    
            except Exception as e:
                self.root.after(0, lambda: self.on_enrollment_error(name, str(e)))
        
        thread = threading.Thread(target=enrollment_worker, daemon=True)
        thread.start()
    
    def start_folder_enrollment(self):
        """Start folder enrollment"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        name = self.folder_name_var.get().strip()
        folder_path = self.files_path_var.get().strip()
        
        
        if not name:
            messagebox.showerror("Error", "Please enter a user name!")
            return
        
        if not folder_path or not os.path.exists(folder_path):
            messagebox.showerror("Error", "Please select a valid folder!")
            return
        
        def enrollment_worker():
            try:
                self.log_to_enrollment(f"🔄 Starting folder enrollment for '{name}'...")
                self.log_to_enrollment(f"📁 Processing images from: {folder_path}")
                
                success = self.system.enroll_person(
                    name=name,
                    images_path=folder_path
                )
                
                if success:
                    self.root.after(0, lambda: self.on_enrollment_success(name))
                else:
                    self.root.after(0, lambda: self.on_enrollment_failed(name))
                    
            except Exception as e:
                self.root.after(0, lambda: self.on_enrollment_error(name, str(e)))
        
        thread = threading.Thread(target=enrollment_worker, daemon=True)
        thread.start()
    
    def on_enrollment_success(self, name):
        """Called when enrollment succeeds"""
        self.log_to_enrollment(f"✅ Successfully enrolled {name}!")
        self.log_to_enrollment("📊 All classifiers retrained with new user data")
        self.webcam_name_var.set("")
        self.folder_name_var.set("")
        self.files_path_var.set("")
        self.refresh_dashboard()
        self.refresh_users_list()
        messagebox.showinfo("Success", f"Successfully enrolled {name}!")
    
    def on_enrollment_failed(self, name):
        """Called when enrollment fails"""
        self.log_to_enrollment(f"❌ Failed to enroll {name}")
        messagebox.showerror("Error", f"Failed to enroll {name}")
    
    def on_enrollment_error(self, name, error):
        """Called when enrollment encounters an error"""
        self.log_to_enrollment(f"❌ Enrollment error for {name}: {error}")
        messagebox.showerror("Error", f"Enrollment error: {error}")
    
    def log_to_enrollment(self, message):
        """Log message to enrollment tab"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.enrollment_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.enrollment_log.see(tk.END)
    
    def log_to_attendance(self, message):
        """Log message to attendance tab"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.attendance_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.attendance_log.see(tk.END)
    
    def browse_folder(self):
        """Browse for image folder"""
        folder_path = filedialog.askdirectory(title="Select Images Folder")
        if folder_path:
            self.files_path_var.set(folder_path)

    def browse_files(self):
        file_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
        if file_paths:
            self.files_path_var.set("; ".join(file_paths))
            self.selected_files = list(file_paths)

    def start_file_enrollment(self):
        """Start enrollment from selected image files"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        if not hasattr(self, "selected_files") or not self.selected_files:
            messagebox.showerror("Error", "Please select image files first!")
            return
        
        # Get user name from the folder enrollment field (reusing the same field)
        name = self.folder_name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a user name!")
            return
        
        def enrollment_worker():
            try:
                self.log_to_enrollment(f"🔄 Starting file enrollment for '{name}'...")
                self.log_to_enrollment(f"📁 Processing {len(self.selected_files)} selected image(s)...")
                
                success = self.system.enroll_person(
                    name=name,
                    file_list=self.selected_files
                )
                
                if success:
                    self.root.after(0, lambda: self.on_enrollment_success(name))
                else:
                    self.root.after(0, lambda: self.on_enrollment_failed(name))
                    
            except Exception as e:
                self.root.after(0, lambda: self.on_enrollment_error(name, str(e)))
        
        thread = threading.Thread(target=enrollment_worker, daemon=True)
        thread.start()

    
    def start_live_attendance(self):
        """Start live attendance monitoring"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        status = self.system.get_system_status()
        if status['enrolled_users'] == 0:
            messagebox.showerror("Error", "No users enrolled! Please enroll users first.")
            return
        
        if self.attendance_running:
            return
        
        self.attendance_running = True
        self.start_attendance_btn.config(state=tk.DISABLED)
        self.stop_attendance_btn.config(state=tk.NORMAL)
        self.attendance_status_var.set("🎥 Live attendance monitoring active")
        
        def attendance_worker():
            try:
                self.log_to_attendance("🎥 Starting user-friendly live attendance...")
                self.log_to_attendance("📊 Multi-classifier system active")
                self.log_to_attendance("💡 Look at the camera - attendance will be marked automatically")
                self.log_to_attendance("🎮 Camera window stays open - press 'q' to quit")
                
                self.system.run_live_attendance()
                
                self.root.after(0, self.on_attendance_stopped)
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.on_attendance_error(error_msg))
        
        thread = threading.Thread(target=attendance_worker, daemon=True)
        thread.start()
    
    def stop_live_attendance(self):
        """Stop live attendance monitoring"""
        self.attendance_running = False
        self.log_to_attendance("⏹️ Stopping attendance monitoring...")
        # Note: The actual stopping is handled by pressing 'q' in the camera window
    
    def on_attendance_stopped(self):
        """Called when attendance monitoring stops"""
        self.attendance_running = False
        self.start_attendance_btn.config(state=tk.NORMAL)
        self.stop_attendance_btn.config(state=tk.DISABLED)
        self.attendance_status_var.set("⏹️ Attendance monitoring stopped")
        self.log_to_attendance("✅ Live attendance session ended")
        self.refresh_dashboard()
    
    def on_attendance_error(self, error):
        """Called when attendance encounters an error"""
        self.attendance_running = False
        self.start_attendance_btn.config(state=tk.NORMAL)
        self.stop_attendance_btn.config(state=tk.DISABLED)
        self.attendance_status_var.set("❌ Attendance error")
        
        # Provide more specific error messages for common issues
        error_msg = str(error)
        if "torch.cat" in error_msg and "non-empty list" in error_msg:
            user_friendly_msg = "Camera/face detection error: No faces detected or invalid image. Please ensure:\n• Camera is working properly\n• Adequate lighting\n• Face is clearly visible\n• Try restarting the camera"
            self.log_to_attendance(f"❌ Face detection error: {user_friendly_msg}")
            messagebox.showerror("Camera Error", user_friendly_msg)
        else:
            self.log_to_attendance(f"❌ Attendance error: {error_msg}")
            messagebox.showerror("Attendance Error", f"Attendance error: {error_msg}")
    
    def refresh_users_list(self):
        """Refresh the users list"""
        if not self.is_initialized:
            return
        
        try:
            # Clear existing items
            for item in self.users_tree.get_children():
                self.users_tree.delete(item)
            
            users = self.system.get_enrolled_users()
            
            # Update total users count
            total_users = len(users)
            self.total_users_label.config(text=f"Total Users: {total_users}")
            
            for _, user in users.iterrows():
                enrollment_date = str(user['enrollment_date']).split()[0]
                last_updated = str(user['last_updated']).split()[0]
                
                self.users_tree.insert('', 'end', values=(
                    user['name'],
                    user['total_embeddings'],
                    enrollment_date,
                    last_updated
                ))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh users list: {e}")
    
    def delete_selected_user(self):
        """Delete selected user"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a user to delete!")
            return
        
        item = self.users_tree.item(selection[0])
        user_name = item['values'][0]
        
        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete user '{user_name}'?\n\nThis will remove all their data and cannot be undone."
        )
        
        if result:
            try:
                success, message = self.system.delete_enrolled_user(user_name)
                if success:
                    messagebox.showinfo("Success", message)
                    self.refresh_users_list()
                    self.refresh_dashboard()
                else:
                    messagebox.showerror("Error", message)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete user: {e}")
    
    def view_user_details(self):
        """View details of selected user"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a user to view details!")
            return
        
        item = self.users_tree.item(selection[0])
        user_name = item['values'][0]
        
        try:
            users = self.system.get_enrolled_users()
            user_row = users[users['name'] == user_name]
            
            if user_row.empty:
                messagebox.showerror("Error", "User not found!")
                return
            
            user = user_row.iloc[0]
            
            details = f"""User Details for: {user['name']}
{'='*50}

Enrollment Information:
• Name: {user['name']}
• Total Face Samples: {user['total_embeddings']}
• Enrollment Date: {user['enrollment_date']}
• Last Updated: {user['last_updated']}

"""
            
            # Get attendance history
            try:
                report = self.system.generate_report()
                user_attendance = report[report['name'] == user_name]
                
                if not user_attendance.empty:
                    details += f"Recent Attendance History ({len(user_attendance)} total records):\n"
                    details += "-" * 50 + "\n"
                    
                    for _, record in user_attendance.head(10).iterrows():
                        timestamp = record['timestamp']
                        confidence = record['confidence']
                        classifier = record.get('classifier_type', 'unknown')
                        details += f"• {timestamp} (confidence: {confidence:.3f}) [{classifier}]\n"
                    
                    if len(user_attendance) > 10:
                        details += f"... and {len(user_attendance) - 10} more records\n"
                else:
                    details += "No attendance records found.\n"
                    
            except Exception as e:
                details += f"Error retrieving attendance history: {e}\n"
            
            self.user_details_text.delete(1.0, tk.END)
            self.user_details_text.insert(tk.END, details)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get user details: {e}")
    
    def generate_report(self):
        """Generate attendance report"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        start_date = self.start_date_var.get().strip() or None
        end_date = self.end_date_var.get().strip() or None
        
        try:
            report = self.system.generate_report(start_date, end_date)
            
            if report.empty:
                self.report_text.delete(1.0, tk.END)
                self.report_text.insert(tk.END, "No attendance records found for the specified period.")
                return
            
            # Generate report text
            report_content = f"""ATTENDANCE REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Date Range: {start_date or 'Beginning'} to {end_date or 'Today'}

SUMMARY:
• Total Records: {len(report)}
• Unique People: {report['name'].nunique()}
• Data Range: {report['date'].min()} to {report['date'].max()}

RECENT RECORDS (Last 20):
{'-'*60}
"""
            
            for _, record in report.head(20).iterrows():
                report_content += f"{record['name']:<20} {record['timestamp']:<20} {record['confidence']:.3f}\n"
            
            if len(report) > 20:
                report_content += f"\n... and {len(report) - 20} more records\n"
            
            # Daily summary
            daily_counts = report['date'].value_counts().sort_index(ascending=False)
            
            report_content += f"\nDAILY SUMMARY (Last 10 days):\n{'-'*60}\n"
            for date_str, count in daily_counts.head(10).items():
                unique_count = report[report['date'] == date_str]['name'].nunique()
                report_content += f"{date_str}: {count} check-ins, {unique_count} unique people\n"
            
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(tk.END, report_content)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")
    
    def export_csv(self):
        """Export attendance report to CSV"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        start_date = self.start_date_var.get().strip() or None
        end_date = self.end_date_var.get().strip() or None
        
        try:
            report = self.system.generate_report(start_date, end_date)
            
            if report.empty:
                messagebox.showwarning("Warning", "No data to export!")
                return
            
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                title="Save Attendance Report",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                report.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Report exported to: {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {e}")
    
    def show_performance_report(self):
        """Show classifier performance report"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        try:
            perf_report = self.system.get_classifier_performance_report()
            
            # Create new window for performance report
            perf_window = tk.Toplevel(self.root)
            perf_window.title("🎯 Classifier Performance Report")
            perf_window.geometry("600x400")
            
            text_widget = scrolledtext.ScrolledText(perf_window, font=('Courier', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(tk.END, perf_report)
            
            # Save button
            def save_perf_report():
                filename = filedialog.asksaveasfilename(
                    title="Save Performance Report",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                )
                if filename:
                    with open(filename, 'w') as f:
                        f.write(perf_report)
                    messagebox.showinfo("Success", f"Performance report saved to: {filename}")
            
            ttk.Button(perf_window, text="💾 Save Report", command=save_perf_report).pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show performance report: {e}")
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive classifier evaluation with multiple metrics"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        # Check if we have enough data
        status = self.system.get_system_status()
        if status['enrolled_users'] < 2:
            messagebox.showerror("Error", "Need at least 2 enrolled users for evaluation!")
            return
        
        if status['total_embeddings'] < 10:
            messagebox.showerror("Error", "Need at least 10 samples for comprehensive evaluation!")
            return
        
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("🔄 Running Comprehensive Evaluation")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (progress_window.winfo_screenheight() // 2) - (150 // 2)
        progress_window.geometry(f"400x150+{x}+{y}")
        
        progress_label = tk.Label(progress_window, text="🔄 Initializing evaluation...", 
                                font=('Arial', 12))
        progress_label.pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()
        
        def run_evaluation():
            try:
                # Update progress
                progress_window.after(0, lambda: progress_label.config(text="📊 Running classifier evaluation..."))
                
                # Run the comprehensive evaluation
                results = self.system.comprehensive_classifier_evaluation(
                    test_size=0.3, 
                    cv_folds=5, 
                    save_plots=True
                )
                
                # Close progress window
                progress_window.after(0, progress_window.destroy)
                
                if 'error' in results:
                    self.root.after(0, lambda: messagebox.showerror("Error", results['error']))
                    return
                
                # Show results in a new window
                self.root.after(0, lambda: self.show_evaluation_results(results))
                
            except Exception as e:
                error_msg = str(e)
                progress_window.after(0, progress_window.destroy)
                self.root.after(0, lambda: messagebox.showerror("Error", f"Evaluation failed: {error_msg}"))
        
        # Run evaluation in background thread
        import threading
        eval_thread = threading.Thread(target=run_evaluation, daemon=True)
        eval_thread.start()
    
    def retrain_and_evaluate(self):
        """Retrain all algorithms with current enrolled students only"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        # Get current system status
        status = self.system.get_system_status()
        current_students = status['enrolled_users']
        total_embeddings = status['total_embeddings_in_db']
        
        if current_students < 2:
            messagebox.showerror("Error", "Need at least 2 enrolled students for evaluation!")
            return
        
        # Show confirmation dialog
        result = messagebox.askyesno(
            "🔄 Retrain & Evaluate Current Students", 
            f"Current enrolled students: {current_students}\n"
            f"Total embeddings: {total_embeddings}\n\n"
            f"This will:\n"
            f"• Retrain all 3 algorithms (SVM, KNN, Logistic Regression)\n"
            f"• Use only current enrolled students (no new data)\n"
            f"• Run comprehensive evaluation with train/test split\n"
            f"• Generate performance metrics and charts\n\n"
            f"Estimated time: 2-5 minutes\n\n"
            f"Continue?"
        )
        
        if not result:
            return
        
        # Start retraining with current data only
        self._run_retrain_current_students_only()
    
    def _run_retrain_current_students_only(self):
        """Run retraining with current enrolled students only (no new data)"""
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("🔄 Retraining Current Students")
        progress_window.geometry("600x200")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (progress_window.winfo_screenheight() // 2) - (200 // 2)
        progress_window.geometry(f"600x200+{x}+{y}")
        
        progress_label = tk.Label(progress_window, text="🔄 Starting retraining with current students...", 
                                font=('Arial', 12))
        progress_label.pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()
        
        def run_retrain():
            try:
                progress_window.after(0, lambda: progress_label.config(text="📊 Rebuilding embeddings from database..."))
                
                # Rebuild embeddings from database (with memory optimization)
                self.system._rebuild_embeddings_from_database(max_images_per_person=10)
                
                progress_window.after(0, lambda: progress_label.config(text="🔄 Retraining all classifiers..."))
                
                # Retrain all classifiers with current data
                self.system._update_all_classifiers_with_new_data()
                
                progress_window.after(0, lambda: progress_label.config(text="💾 Saving updated models..."))
                
                # Save the updated system
                self.system.save_system()
                
                progress_window.after(0, lambda: progress_label.config(text="📊 Running comprehensive evaluation..."))
                
                # Run comprehensive evaluation with train/test split
                results = self.system.train_test_evaluate_and_save(
                    test_size=0.2, 
                    random_state=42, 
                    output_to_files=True
                )
                
                progress_window.after(0, progress_window.destroy)
                
                if 'error' in results:
                    self.root.after(0, lambda: messagebox.showerror("Error", results['error']))
                    return
                
                # Show results
                self.root.after(0, lambda: self.show_retrain_evaluation_results(results))
                
            except Exception as e:
                error_msg = str(e)
                progress_window.after(0, progress_window.destroy)
                self.root.after(0, lambda: messagebox.showerror("Error", f"Retraining failed: {error_msg}"))
        
        # Run retraining in background thread
        import threading
        retrain_thread = threading.Thread(target=run_retrain, daemon=True)
        retrain_thread.start()
    
    def generate_performance_charts(self):
        """Generate performance charts from latest evaluation results"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        # Check if we have evaluation results
        models_dir = Path('models')
        csv_files = list(models_dir.glob('algorithm_comparison_*.csv'))
        
        if not csv_files:
            messagebox.showerror("Error", "No evaluation results found!\nPlease run 'Retrain & Evaluate' first.")
            return
        
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("📊 Generating Performance Charts")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (progress_window.winfo_screenheight() // 2) - (150 // 2)
        progress_window.geometry(f"400x150+{x}+{y}")
        
        progress_label = tk.Label(progress_window, text="📊 Loading evaluation results...", 
                                font=('Arial', 12))
        progress_label.pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()
        
        def generate_charts():
            try:
                # Update progress
                progress_window.after(0, lambda: progress_label.config(text="📊 Creating performance charts..."))
                
                # Load latest results
                latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(latest_csv)
                
                # Load metadata
                comprehensive_file = models_dir / 'comprehensive_evaluation_results.json'
                metadata = {}
                if comprehensive_file.exists():
                    with open(comprehensive_file, 'r') as f:
                        data = json.load(f)
                        metadata = data.get('training_metadata', {})
                
                # Generate charts
                progress_window.after(0, lambda: progress_label.config(text="📊 Rendering charts..."))
                
                # Create the charts
                fig = self._create_performance_charts(df, metadata)
                
                # Close progress window
                progress_window.after(0, progress_window.destroy)
                
                # Show charts in new window
                self.root.after(0, lambda: self._show_charts_window(fig, df, metadata))
                
            except Exception as e:
                error_msg = str(e)
                progress_window.after(0, progress_window.destroy)
                self.root.after(0, lambda: messagebox.showerror("Error", f"Chart generation failed: {error_msg}"))
        
        # Run chart generation in background thread
        import threading
        chart_thread = threading.Thread(target=generate_charts, daemon=True)
        chart_thread.start()
    
    def _create_performance_charts(self, df, metadata):
        """Create performance charts (internal method)"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure
        fig = plt.figure(figsize=(22, 18))
        
        # Define colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # Chart 1: Main Performance Metrics
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = df[metric] * 100
            bars = ax1.bar(x + i*width, values, width, 
                          label=metric.replace('_', ' ').title(), 
                          color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1.0,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax1.set_xlabel('Algorithms', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Performance (%)', fontsize=11, fontweight='bold')
        ax1.set_title('📊 Algorithm Performance Comparison\n(Main Metrics)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(df['classifier'], fontsize=10, rotation=0)
        ax1.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
        
        # Chart 2: Training Time
        ax2 = plt.subplot(2, 3, 2)
        training_times = df['training_time'] * 1000
        bars2 = ax2.bar(df['classifier'], training_times, color=colors[:len(df)], alpha=0.8)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Algorithms', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Training Time (ms)', fontsize=11, fontweight='bold')
        ax2.set_title('⚡ Training Time Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticklabels(df['classifier'], fontsize=10, rotation=0)
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Inference Speed
        ax3 = plt.subplot(2, 3, 3)
        inference_speeds = df['inference_speed']
        bars3 = ax3.bar(df['classifier'], inference_speeds, color=colors[:len(df)], alpha=0.8)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.set_xlabel('Algorithms', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Inference Speed (samples/sec)', fontsize=11, fontweight='bold')
        ax3.set_title('🚀 Inference Speed Comparison', fontsize=12, fontweight='bold')
        ax3.set_xticklabels(df['classifier'], fontsize=10, rotation=0)
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Accuracy Focus
        ax4 = plt.subplot(2, 3, 4)
        accuracy_values = df['accuracy'] * 100
        bars4 = ax4.bar(df['classifier'], accuracy_values, color=colors[:len(df)], alpha=0.8)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4.set_xlabel('Algorithms', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax4.set_title('🎯 Accuracy Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticklabels(df['classifier'], fontsize=10, rotation=0)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        # Chart 5: F1-Score
        ax5 = plt.subplot(2, 3, 5)
        f1_values = df['f1_score'] * 100
        bars5 = ax5.bar(df['classifier'], f1_values, color=colors[:len(df)], alpha=0.8)
        
        for bar in bars5:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax5.set_xlabel('Algorithms', fontsize=11, fontweight='bold')
        ax5.set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
        ax5.set_title('⚖️ F1-Score Comparison', fontsize=12, fontweight='bold')
        ax5.set_xticklabels(df['classifier'], fontsize=10, rotation=0)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 100)
        
        # Chart 6: Overall Score
        ax6 = plt.subplot(2, 3, 6)
        weights = {'accuracy': 0.3, 'precision': 0.25, 'recall': 0.25, 'f1_score': 0.2}
        overall_scores = []
        
        for _, row in df.iterrows():
            score = sum(row[metric] * weight for metric, weight in weights.items())
            overall_scores.append(score * 100)
        
        bars6 = ax6.bar(df['classifier'], overall_scores, color=colors[:len(df)], alpha=0.8)
        
        for bar in bars6:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax6.set_xlabel('Algorithms', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Overall Score (%)', fontsize=11, fontweight='bold')
        ax6.set_title('🏆 Overall Performance Score\n(Weighted Average)', fontsize=12, fontweight='bold')
        ax6.set_xticklabels(df['classifier'], fontsize=10, rotation=0)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 100)
        
        # Add dataset info
        dataset_info = f"""📊 Dataset: {metadata.get('total_people', 'N/A')} people, {metadata.get('total_embeddings', 'N/A')} embeddings
🎯 Test Split: 20% (Stratified) | Date: {metadata.get('training_date', 'N/A')}"""
        
        fig.text(0.02, 0.02, dataset_info, fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Add best performers
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        best_f1 = df.loc[df['f1_score'].idxmax()]
        fastest = df.loc[df['inference_speed'].idxmax()]
        
        summary = f"""🏆 Best: {best_accuracy['classifier']} ({best_accuracy['accuracy']*100:.1f}% acc)
⚡ Fastest: {fastest['classifier']} ({fastest['inference_speed']:.1f} sps)"""
        
        fig.text(0.98, 0.02, summary, fontsize=10, ha='right',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=2.0)
        plt.subplots_adjust(bottom=0.20, top=0.92, left=0.06, right=0.94, hspace=0.4, wspace=0.3)
        
        return fig
    
    def _show_summary_only(self, df, metadata):
        """Show only summary table when charts fail"""
        try:
            # Create simple summary window
            summary_window = tk.Toplevel(self.root)
            summary_window.title("📊 Algorithm Performance Summary")
            summary_window.geometry("800x600")
            
            summary_text = scrolledtext.ScrolledText(summary_window, font=('Courier', 10))
            summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create summary content
            summary_content = "📊 ALGORITHM PERFORMANCE SUMMARY\n"
            summary_content += "=" * 80 + "\n\n"
            
            for _, row in df.iterrows():
                summary_content += f"🔹 {row['classifier']}\n"
                summary_content += f"   Accuracy: {row['accuracy']*100:.2f}%\n"
                summary_content += f"   Precision: {row['precision']*100:.2f}%\n"
                summary_content += f"   Recall: {row['recall']*100:.2f}%\n"
                summary_content += f"   F1-Score: {row['f1_score']*100:.2f}%\n"
                summary_content += f"   Training Time: {row['training_time']*1000:.2f}ms\n"
                summary_content += f"   Inference Time: {row['inference_time']*1000:.2f}ms\n"
                summary_content += f"   Inference Speed: {row['inference_speed']:.1f} faces/sec\n\n"
            
            summary_content += f"\n📈 Dataset Info:\n"
            summary_content += f"   Total Samples: {metadata.get('total_samples', 'N/A')}\n"
            summary_content += f"   Total People: {metadata.get('total_people', 'N/A')}\n"
            summary_content += f"   Train/Test Split: {metadata.get('test_size', 'N/A')}\n"
            summary_content += f"   Evaluation Date: {metadata.get('evaluation_date', 'N/A')}\n"
            
            summary_text.insert(tk.END, summary_content)
            summary_text.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"Summary display error: {e}")
            messagebox.showerror("Error", f"Failed to display performance summary: {e}")
    
    def _show_charts_window(self, fig, df, metadata):
        """Show charts in a new window with save options"""
        try:
            # Create new window
            charts_window = tk.Toplevel(self.root)
            charts_window.title("📊 Algorithm Performance Charts")
            charts_window.geometry("1600x1000")
            
            # Create notebook for different views
            notebook = ttk.Notebook(charts_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Tab 1: Charts View
            charts_frame = ttk.Frame(notebook)
            notebook.add(charts_frame, text="📊 Performance Charts")
            
            # Save chart as image and display it (safer approach)
            import tempfile
            import os
            from PIL import Image, ImageTk
            
            # Save figure to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig.savefig(temp_file.name, dpi=100, bbox_inches='tight')
            temp_file.close()
            
            # Load and display image
            img = Image.open(temp_file.name)
            img = img.resize((1400, 800), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Create label to display image
            img_label = tk.Label(charts_frame, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack(fill=tk.BOTH, expand=True)
            
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass
            
            # Handle window close properly
            def on_closing():
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    charts_window.destroy()
                except:
                    pass
            
            charts_window.protocol("WM_DELETE_WINDOW", on_closing)
            
        except Exception as e:
            print(f"Chart display error: {e}")
            # Fallback: just show the summary table
            self._show_summary_only(df, metadata)
            return
        
        # Tab 2: Summary Table
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📋 Summary Table")
        
        summary_text = scrolledtext.ScrolledText(summary_frame, font=('Courier', 10))
        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create summary table
        summary_content = "📊 ALGORITHM PERFORMANCE SUMMARY\n"
        summary_content += "=" * 80 + "\n\n"
        
        # Format dataframe for display
        display_df = df.copy()
        for col in ['accuracy', 'precision', 'recall', 'f1_score']:
            display_df[col] = (display_df[col] * 100).round(2).astype(str) + '%'
        
        display_df['training_time'] = (display_df['training_time'] * 1000).round(2).astype(str) + 'ms'
        display_df['inference_time'] = display_df['inference_time'].round(3).astype(str) + 's'
        display_df['inference_speed'] = display_df['inference_speed'].round(1).astype(str) + ' sps'
        
        summary_content += display_df.to_string(index=False)
        
        summary_content += "\n\n🏆 RANKINGS:\n"
        summary_content += "-" * 40 + "\n"
        summary_content += f"🥇 Best Accuracy: {df.loc[df['accuracy'].idxmax(), 'classifier']} ({df['accuracy'].max()*100:.2f}%)\n"
        summary_content += f"🥇 Best Precision: {df.loc[df['precision'].idxmax(), 'classifier']} ({df['precision'].max()*100:.2f}%)\n"
        summary_content += f"🥇 Best Recall: {df.loc[df['recall'].idxmax(), 'classifier']} ({df['recall'].max()*100:.2f}%)\n"
        summary_content += f"🥇 Best F1-Score: {df.loc[df['f1_score'].idxmax(), 'classifier']} ({df['f1_score'].max()*100:.2f}%)\n"
        summary_content += f"⚡ Fastest Training: {df.loc[df['training_time'].idxmin(), 'classifier']} ({df['training_time'].min()*1000:.2f}ms)\n"
        summary_content += f"🚀 Fastest Inference: {df.loc[df['inference_speed'].idxmax(), 'classifier']} ({df['inference_speed'].max():.1f} samples/sec)\n"
        
        summary_content += f"\n📊 DATASET INFORMATION:\n"
        summary_content += f"• Total People: {metadata.get('total_people', 'N/A')}\n"
        summary_content += f"• Total Embeddings: {metadata.get('total_embeddings', 'N/A')}\n"
        summary_content += f"• Max Images per Person: {metadata.get('max_images_per_person', 'N/A')}\n"
        summary_content += f"• Training Date: {metadata.get('training_date', 'N/A')}\n"
        
        summary_text.insert(tk.END, summary_content)
        
        # Button frame
        btn_frame = ttk.Frame(charts_window)
        btn_frame.pack(pady=10)
        
        # Save buttons
        ttk.Button(btn_frame, text="💾 Save Charts as PNG", 
                  command=lambda: self._save_charts(fig, 'png')).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="📄 Save Charts as PDF", 
                  command=lambda: self._save_charts(fig, 'pdf')).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="✅ Close", 
                  command=charts_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def _save_charts(self, fig, format_type):
        """Save charts to file"""
        from tkinter import filedialog
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"algorithm_performance_charts_{timestamp}.{format_type}"
        
        filename = filedialog.asksaveasfilename(
            defaultextension=f".{format_type}",
            filetypes=[(f"{format_type.upper()} files", f"*.{format_type}")],
            initialvalue=default_filename
        )
        
        if filename:
            try:
                fig.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                messagebox.showinfo("Success", f"Charts saved as: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save charts: {e}")
    
    def add_more_students(self):
        """Add 100 more students to the system"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        # Get current student count
        current_users = self.system.get_enrolled_users()
        current_count = len(current_users)
        
        # Show confirmation dialog
        result = messagebox.askyesno(
            "👥 Add 100 More Students", 
            f"Current students: {current_count}\n"
            f"Adding: 100 more students\n"
            f"Final total: {current_count + 100} students\n\n"
            f"This will:\n"
            f"• Add students from Original Images dataset\n"
            f"• Add students from Faces dataset\n"
            f"• Add students from LFW dataset if needed\n"
            f"• Each student will have exactly 10 samples\n"
            f"• Skip duplicate students (already enrolled)\n"
            f"• Retrain all classifiers\n\n"
            f"Continue?"
        )
        
        if not result:
            return
        
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("👥 Adding 100 More Students")
        progress_window.geometry("600x200")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (progress_window.winfo_screenheight() // 2) - (200 // 2)
        progress_window.geometry(f"600x200+{x}+{y}")
        
        progress_label = tk.Label(progress_window, text="👥 Adding 100 more students from datasets...", 
                                font=('Arial', 12))
        progress_label.pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()
        
        def run_add_students():
            try:
                progress_window.after(0, lambda: progress_label.config(text="📊 Adding students from Original Images..."))
                
                # Add 100 more students
                result = self.system.add_more_students(
                    add_count=100,
                    faces_dir='Faces/Faces',
                    originals_dir='Original Images/Original Images',
                    lfw_path='lfw-funneled/lfw_funneled',
                    max_lfw_people=50
                )
                
                progress_window.after(0, progress_window.destroy)
                
                # Show results
                if result['target_reached']:
                    message = (f"✅ SUCCESS! Added 100 more students!\n\n"
                             f"📊 Previous students: {result['current']}\n"
                             f"📈 Students added: {result['added']}\n"
                             f"📊 Final total: {result['final']}\n\n"
                             f"All students have been enrolled and system retrained!")
                else:
                    message = (f"⚠️ PARTIAL SUCCESS\n\n"
                             f"📊 Previous students: {result['current']}\n"
                             f"📈 Students added: {result['added']}\n"
                             f"📊 Final total: {result['final']}\n\n"
                             f"Could not add 100 students with available datasets.")
                
                self.root.after(0, lambda: messagebox.showinfo("Add Students Complete", message))
                
                # Refresh the users list
                self.root.after(0, self.refresh_users_list)
                
            except Exception as e:
                error_msg = str(e)
                progress_window.after(0, progress_window.destroy)
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to add students: {error_msg}"))
        
        # Run in background thread
        import threading
        add_thread = threading.Thread(target=run_add_students, daemon=True)
        add_thread.start()
    
    def show_system_status(self):
        """Show detailed system status including memory optimization info"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        try:
            status = self.system.get_system_status()
            
            # Create status window
            status_window = tk.Toplevel(self.root)
            status_window.title("📊 System Status & Memory Optimization")
            status_window.geometry("600x500")
            status_window.transient(self.root)
            status_window.grab_set()
            
            # Center the window
            status_window.update_idletasks()
            x = (status_window.winfo_screenwidth() // 2) - (600 // 2)
            y = (status_window.winfo_screenheight() // 2) - (500 // 2)
            status_window.geometry(f"600x500+{x}+{y}")
            
            # Title
            tk.Label(status_window, text="📊 System Status & Memory Optimization", 
                    font=('Arial', 16, 'bold')).pack(pady=10)
            
            # Create scrollable text widget
            text_frame = tk.Frame(status_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
            scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Format status information
            status_text = f"""
🎯 SYSTEM OVERVIEW
{'='*50}
System Type: {status['system_type']}
Enrolled Users: {status['enrolled_users']}
Active Classifiers: {', '.join(status['active_classifiers'])}

📊 DATASET STATISTICS
{'='*50}
Total Embeddings in Database: {status['total_embeddings_in_db']}
Total Embeddings in Memory: {status['total_embeddings']}
Average Images per Person: {status['avg_images_per_person']}
Max Images per Person: {status['max_images_per_person']}
Min Images per Person: {status['min_images_per_person']}

🧠 MEMORY OPTIMIZATION
{'='*50}
Memory Optimized: {'✅ YES' if status['memory_optimized'] else '❌ NO'}
Optimization Status: {'Active (5 images per person limit)' if status['memory_optimized'] else 'Not needed'}

⚙️ SYSTEM SETTINGS
{'='*50}
Confidence Threshold: {status['confidence_threshold']}
Verification Threshold: {status['verification_threshold']}

🔧 PERFORMANCE INFO
{'='*50}
"""
            
            # Add classifier performance if available
            if status['classifier_performance']:
                status_text += "Classifier Performance:\n"
                for clf_name, perf in status['classifier_performance'].items():
                    status_text += f"  {clf_name}: {perf.get('accuracy', 'N/A')} accuracy\n"
            else:
                status_text += "Classifier Performance: Not available (run evaluation first)\n"
            
            # Add recommendations
            status_text += f"""
💡 RECOMMENDATIONS
{'='*50}
"""
            
            if status['total_embeddings_in_db'] > 1000:
                status_text += "⚠️  Large dataset detected - memory optimization is active\n"
                status_text += "✅ Training should be faster with optimized dataset\n"
            else:
                status_text += "✅ Dataset size is manageable - no optimization needed\n"
            
            if status['max_images_per_person'] > 10:
                status_text += f"⚠️  Some users have {status['max_images_per_person']} images - consider limiting\n"
            
            if status['enrolled_users'] < 10:
                status_text += "💡 Consider adding more users for better model performance\n"
            
            # Insert text
            text_widget.insert(tk.END, status_text)
            text_widget.config(state=tk.DISABLED)
            
            # Buttons
            btn_frame = tk.Frame(status_window)
            btn_frame.pack(pady=10)
            
            ttk.Button(btn_frame, text="🔄 Refresh Status", 
                      command=lambda: [status_window.destroy(), self.show_system_status()]).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="❌ Close", 
                      command=status_window.destroy).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get system status: {e}")
    
    def cleanup_low_samples(self):
        """Remove students with less than 10 sample images"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        # Get current system status
        status = self.system.get_system_status()
        current_students = status['enrolled_users']
        min_samples = status['min_images_per_person']
        
        if current_students == 0:
            messagebox.showerror("Error", "No enrolled students found!")
            return
        
        # Show configuration dialog
        config_window = tk.Toplevel(self.root)
        config_window.title("🧹 Clean Up Low Sample Students")
        config_window.geometry("500x400")
        config_window.transient(self.root)
        config_window.grab_set()
        
        # Center the window
        config_window.update_idletasks()
        x = (config_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (config_window.winfo_screenheight() // 2) - (400 // 2)
        config_window.geometry(f"500x400+{x}+{y}")
        
        # Configuration options
        tk.Label(config_window, text="🧹 Clean Up Low Sample Students", font=('Arial', 16, 'bold')).pack(pady=10)
        
        tk.Label(config_window, text=f"Current students: {current_students}", font=('Arial', 12)).pack(pady=5)
        tk.Label(config_window, text=f"Minimum samples per student: {min_samples}", font=('Arial', 12)).pack(pady=5)
        
        tk.Label(config_window, text="\nThis will remove students with insufficient sample images:", font=('Arial', 12)).pack(pady=5)
        tk.Label(config_window, text="• Students with less than minimum samples will be deleted", font=('Arial', 10)).pack()
        tk.Label(config_window, text="• System will be retrained with remaining students", font=('Arial', 10)).pack()
        tk.Label(config_window, text="• This action cannot be undone", font=('Arial', 10, 'bold'), fg='red').pack()
        
        # Minimum samples setting
        tk.Label(config_window, text="\nMinimum Samples Required:", font=('Arial', 12)).pack(pady=5)
        min_samples_var = tk.StringVar(value="10")
        min_samples_entry = tk.Entry(config_window, textvariable=min_samples_var, width=20)
        min_samples_entry.pack(pady=5)
        tk.Label(config_window, text="(Students with less than this will be removed)", font=('Arial', 9)).pack()
        
        # Warning
        warning_text = """
⚠️ This action will permanently delete students with insufficient samples
• Students with less than minimum samples will be removed
• System will be retrained automatically
• This cannot be undone
        """
        tk.Label(config_window, text=warning_text, font=('Arial', 10), 
                fg='red', justify='left').pack(pady=10)
        
        def start_cleanup():
            try:
                min_samples = int(min_samples_var.get())
                
                if min_samples < 1:
                    messagebox.showerror("Error", "Minimum samples must be at least 1!")
                    return
                
                config_window.destroy()
                self._run_cleanup_low_samples(min_samples)
                
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number!")
        
        # Buttons
        btn_frame = tk.Frame(config_window)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="🧹 Start Cleanup", 
                  command=start_cleanup).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="❌ Cancel", 
                  command=config_window.destroy).pack(side=tk.LEFT, padx=10)
    
    def _run_cleanup_low_samples(self, min_samples):
        """Run cleanup of low sample students in background"""
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("🧹 Cleaning Up Low Sample Students")
        progress_window.geometry("600x200")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (progress_window.winfo_screenheight() // 2) - (200 // 2)
        progress_window.geometry(f"600x200+{x}+{y}")
        
        progress_label = tk.Label(progress_window, text="🧹 Starting cleanup of low sample students...", 
                                font=('Arial', 12))
        progress_label.pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()
        
        def run_cleanup():
            try:
                progress_window.after(0, lambda: progress_label.config(text="🔍 Checking students for minimum samples..."))
                
                # Remove students with insufficient samples
                results = self.system.remove_students_with_insufficient_samples(min_samples)
                
                progress_window.after(0, progress_window.destroy)
                
                if 'error' in results:
                    self.root.after(0, lambda: messagebox.showerror("Error", results['error']))
                    return
                
                # Show results
                self.root.after(0, lambda: self.show_cleanup_results(results))
                
            except Exception as e:
                error_msg = str(e)
                progress_window.after(0, progress_window.destroy)
                self.root.after(0, lambda: messagebox.showerror("Error", f"Cleanup failed: {error_msg}"))
        
        # Run cleanup in background thread
        import threading
        cleanup_thread = threading.Thread(target=run_cleanup, daemon=True)
        cleanup_thread.start()
    
    def show_cleanup_results(self, results):
        """Display cleanup results"""
        # Create new window for results
        results_window = tk.Toplevel(self.root)
        results_window.title("🧹 Cleanup Results")
        results_window.geometry("800x600")
        results_window.transient(self.root)
        results_window.grab_set()
        
        # Center the window
        results_window.update_idletasks()
        x = (results_window.winfo_screenwidth() // 2) - (800 // 2)
        y = (results_window.winfo_screenheight() // 2) - (600 // 2)
        results_window.geometry(f"800x600+{x}+{y}")
        
        # Title
        title_label = tk.Label(results_window, text="🧹 Cleanup Results", 
                             font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Summary
        summary_text = f"""
Cleanup Summary:
• Minimum samples required: {results['min_samples']}
• Students removed: {results['removed_count']}
• Students kept: {results['kept_count']}
• Total students before: {results['removed_count'] + results['kept_count']}
• Total students after: {results['kept_count']}
        """
        
        summary_label = tk.Label(results_window, text=summary_text, 
                               font=('Arial', 12), justify='left')
        summary_label.pack(pady=10, padx=20)
        
        # Create notebook for detailed results
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Removed students tab
        if results['removed_students']:
            removed_frame = ttk.Frame(notebook)
            notebook.add(removed_frame, text=f"Removed Students ({results['removed_count']})")
            
            removed_text = tk.Text(removed_frame, wrap=tk.WORD, height=15)
            removed_scrollbar = ttk.Scrollbar(removed_frame, orient=tk.VERTICAL, command=removed_text.yview)
            removed_text.configure(yscrollcommand=removed_scrollbar.set)
            
            removed_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            removed_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            removed_content = "Removed Students:\n" + "="*50 + "\n"
            for student in results['removed_students']:
                removed_content += f"• {student['name']}: {student['samples']} samples\n"
            
            removed_text.insert(tk.END, removed_content)
            removed_text.config(state=tk.DISABLED)
        
        # Kept students tab
        if results['kept_students']:
            kept_frame = ttk.Frame(notebook)
            notebook.add(kept_frame, text=f"Kept Students ({results['kept_count']})")
            
            kept_text = tk.Text(kept_frame, wrap=tk.WORD, height=15)
            kept_scrollbar = ttk.Scrollbar(kept_frame, orient=tk.VERTICAL, command=kept_text.yview)
            kept_text.configure(yscrollcommand=kept_scrollbar.set)
            
            kept_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            kept_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            kept_content = "Kept Students:\n" + "="*50 + "\n"
            for student in results['kept_students']:
                kept_content += f"• {student['name']}: {student['samples']} samples\n"
            
            kept_text.insert(tk.END, kept_content)
            kept_text.config(state=tk.DISABLED)
        
        # Close button
        close_btn = ttk.Button(results_window, text="✅ Close", 
                             command=results_window.destroy)
        close_btn.pack(pady=10)
    
    def show_retrain_evaluation_results(self, results):
        """Display retrain evaluation results with enhanced information"""
        # Create new window for results
        results_window = tk.Toplevel(self.root)
        results_window.title("🔄 Retrain & Evaluation Results - All Algorithms")
        results_window.geometry("1200x800")
        
        # Create notebook for different tabs
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Algorithm Comparison
        comparison_frame = ttk.Frame(notebook)
        notebook.add(comparison_frame, text="📊 Algorithm Comparison")
        
        comparison_text = scrolledtext.ScrolledText(comparison_frame, font=('Courier', 10))
        comparison_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate comparison report
        comparison_report = "🔄 RETRAIN & EVALUATION RESULTS\n"
        comparison_report += "=" * 60 + "\n\n"
        comparison_report += f"📅 Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        comparison_report += f"📊 Dataset: {len(self.system.known_embeddings)} embeddings, {len(set(self.system.known_names))} people\n"
        comparison_report += f"🎯 Test Split: 20% held-out for evaluation\n\n"
        
        comparison_report += "📈 PERFORMANCE METRICS:\n"
        comparison_report += "-" * 60 + "\n"
        comparison_report += f"{'Algorithm':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
        comparison_report += "-" * 60 + "\n"
        
        for clf_name, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                comparison_report += f"{clf_name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
                comparison_report += f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}\n"
        
        comparison_report += "\n⚡ COMPUTATIONAL EFFICIENCY:\n"
        comparison_report += "-" * 60 + "\n"
        comparison_report += f"{'Algorithm':<20} {'Training(s)':<12} {'Inference(s)':<12} {'Speed(sps)':<12}\n"
        comparison_report += "-" * 60 + "\n"
        
        for clf_name, metrics in results.items():
            if isinstance(metrics, dict) and 'training_time' in metrics:
                comparison_report += f"{clf_name:<20} {metrics['training_time']:<12.3f} "
                comparison_report += f"{metrics['inference_time']:<12.3f} {metrics['inference_speed']:<12.1f}\n"
        
        # Find best performers
        best_accuracy = max(results.items(), key=lambda x: x[1].get('accuracy', 0) if isinstance(x[1], dict) else 0)
        best_f1 = max(results.items(), key=lambda x: x[1].get('f1_score', 0) if isinstance(x[1], dict) else 0)
        best_speed = max(results.items(), key=lambda x: x[1].get('inference_speed', 0) if isinstance(x[1], dict) else 0)
        
        comparison_report += "\n🏆 BEST PERFORMERS:\n"
        comparison_report += "-" * 40 + "\n"
        if isinstance(best_accuracy[1], dict):
            comparison_report += f"🎯 Best Accuracy:   {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.3f})\n"
        if isinstance(best_f1[1], dict):
            comparison_report += f"🎯 Best F1-Score:   {best_f1[0]} ({best_f1[1]['f1_score']:.3f})\n"
        if isinstance(best_speed[1], dict):
            comparison_report += f"⚡ Fastest:         {best_speed[0]} ({best_speed[1]['inference_speed']:.1f} sps)\n"
        
        comparison_report += "\n💾 SAVED FILES:\n"
        comparison_report += "-" * 40 + "\n"
        comparison_report += "• models/KNN_results.json\n"
        comparison_report += "• models/SVM_results.json\n"
        comparison_report += "• models/LogisticRegression_results.json\n"
        comparison_report += "• models/algorithm_comparison_YYYYMMDD_HHMMSS.csv\n"
        comparison_report += "• models/comprehensive_evaluation_results.json\n"
        
        comparison_text.insert(tk.END, comparison_report)
        
        # Tab 2: Detailed Metrics
        details_frame = ttk.Frame(notebook)
        notebook.add(details_frame, text="📋 Detailed Metrics")
        
        details_text = scrolledtext.ScrolledText(details_frame, font=('Courier', 9))
        details_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for clf_name, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                details_text.insert(tk.END, f"\n{'='*50}\n")
                details_text.insert(tk.END, f"ALGORITHM: {clf_name}\n")
                details_text.insert(tk.END, f"{'='*50}\n")
                details_text.insert(tk.END, f"Accuracy:     {metrics['accuracy']:.4f}\n")
                details_text.insert(tk.END, f"Precision:    {metrics['precision']:.4f}\n")
                details_text.insert(tk.END, f"Recall:       {metrics['recall']:.4f}\n")
                details_text.insert(tk.END, f"F1-Score:     {metrics['f1_score']:.4f}\n")
                details_text.insert(tk.END, f"Training Time: {metrics['training_time']:.3f}s\n")
                details_text.insert(tk.END, f"Inference Time: {metrics['inference_time']:.3f}s\n")
                details_text.insert(tk.END, f"Inference Speed: {metrics['inference_speed']:.1f} samples/sec\n")
                
                if 'class_names' in metrics:
                    details_text.insert(tk.END, f"\nClasses: {', '.join(metrics['class_names'])}\n")
        
        # Tab 3: System Status
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="ℹ️ System Status")
        
        status_text = scrolledtext.ScrolledText(status_frame, font=('Courier', 10))
        status_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        system_status = self.system.get_system_status()
        status_report = "🔄 RETRAINED SYSTEM STATUS\n"
        status_report += "=" * 40 + "\n\n"
        status_report += f"System Type: {system_status['system_type']}\n"
        status_report += f"Enrolled Users: {system_status['enrolled_users']}\n"
        status_report += f"Total Embeddings: {system_status['total_embeddings']}\n"
        status_report += f"Active Classifiers: {', '.join(system_status['active_classifiers'])}\n"
        status_report += f"Confidence Threshold: {system_status['confidence_threshold']}\n"
        status_report += f"Verification Threshold: {system_status['verification_threshold']}\n\n"
        
        status_report += "📊 CLASSIFIER PERFORMANCE TRACKING:\n"
        status_report += "-" * 40 + "\n"
        for clf_name, perf in system_status['classifier_performance'].items():
            if perf['total'] > 0:
                accuracy = (perf['correct'] / perf['total']) * 100
                status_report += f"{clf_name}: {perf['correct']}/{perf['total']} ({accuracy:.1f}%)\n"
            else:
                status_report += f"{clf_name}: No predictions yet\n"
        
        status_text.insert(tk.END, status_report)
        
        # Add close button
        close_btn = ttk.Button(results_window, text="✅ Close", 
                              command=results_window.destroy)
        close_btn.pack(pady=10)
    
    def show_evaluation_results(self, results):
        """Display comprehensive evaluation results"""
        # Create new window for results
        results_window = tk.Toplevel(self.root)
        results_window.title("📊 Comprehensive Classifier Evaluation Results")
        results_window.geometry("1000x700")
        
        # Create notebook for different tabs
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Summary Report
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📋 Summary Report")
        
        summary_text = scrolledtext.ScrolledText(summary_frame, font=('Courier', 10))
        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        summary_text.insert(tk.END, results['comparison_report'])
        
        # Tab 2: Detailed Metrics
        details_frame = ttk.Frame(notebook)
        notebook.add(details_frame, text="📊 Detailed Metrics")
        
        # Create treeview for detailed metrics
        columns = ('Metric', 'SVM', 'KNN', 'LogisticRegression')
        tree = ttk.Treeview(details_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor='center')
        
        # Populate the tree with metrics
        metrics_to_show = [
            ('Accuracy', 'accuracy'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1-Score', 'f1_score'),
            ('CV Accuracy Mean', 'cv_accuracy_mean'),
            ('CV Accuracy Std', 'cv_accuracy_std'),
            ('Training Time (s)', 'training_time'),
            ('Inference Time (s)', 'inference_time'),
            ('Inference Speed (sps)', 'inference_speed')
        ]
        
        for metric_name, metric_key in metrics_to_show:
            values = [metric_name]
            for clf_name in ['SVM', 'KNN', 'LogisticRegression']:
                if clf_name in results and clf_name != 'training_metadata':
                    # Handle missing metrics gracefully
                    if metric_key in results[clf_name]:
                        value = results[clf_name][metric_key]
                        if metric_key in ['training_time', 'inference_time']:
                            values.append(f"{value:.4f}")
                        elif metric_key == 'inference_speed':
                            values.append(f"{value:.1f}")
                        else:
                            values.append(f"{value:.3f}")
                    else:
                        # Handle missing cross-validation metrics
                        if metric_key in ['cv_accuracy_mean', 'cv_accuracy_std']:
                            values.append("N/A")
                        else:
                            values.append("Missing")
                else:
                    values.append("N/A")
            tree.insert('', 'end', values=values)
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollbar to treeview
        scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tab 3: Per-Class Performance
        class_frame = ttk.Frame(notebook)
        notebook.add(class_frame, text="👥 Per-Class Performance")
        
        class_text = scrolledtext.ScrolledText(class_frame, font=('Courier', 10))
        class_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate per-class report
        class_report = "=" * 60 + "\n"
        class_report += "                PER-CLASS PERFORMANCE ANALYSIS\n"
        class_report += "=" * 60 + "\n\n"
        
        for clf_name in ['SVM', 'KNN', 'LogisticRegression']:
            if clf_name in results and clf_name != 'training_metadata':
                class_report += f"{clf_name} CLASSIFIER:\n"
                class_report += "-" * 30 + "\n"
                
                # Check if per-class metrics exist
                if ('class_names' in results[clf_name] and 
                    'precision_per_class' in results[clf_name] and
                    'recall_per_class' in results[clf_name] and
                    'f1_per_class' in results[clf_name]):
                    
                    class_names = results[clf_name]['class_names']
                    precision_per_class = results[clf_name]['precision_per_class']
                    recall_per_class = results[clf_name]['recall_per_class']
                    f1_per_class = results[clf_name]['f1_per_class']
                    
                    class_report += f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
                    class_report += "-" * 50 + "\n"
                    
                    for i, class_name in enumerate(class_names):
                        if i < len(precision_per_class):
                            class_name_short = class_name[:12] if len(class_name) > 12 else class_name
                            class_report += f"{class_name_short:<15} {precision_per_class[i]:<10.3f} "
                            class_report += f"{recall_per_class[i]:<10.3f} {f1_per_class[i]:<10.3f}\n"
                else:
                    # Show overall metrics instead
                    accuracy = results[clf_name].get('accuracy', 0)
                    precision = results[clf_name].get('precision', 0)
                    recall = results[clf_name].get('recall', 0)
                    f1 = results[clf_name].get('f1_score', 0)
                    
                    class_report += f"Overall Performance:\n"
                    class_report += f"  Accuracy:  {accuracy:.3f}\n"
                    class_report += f"  Precision: {precision:.3f}\n"
                    class_report += f"  Recall:    {recall:.3f}\n"
                    class_report += f"  F1-Score:  {f1:.3f}\n"
                
                class_report += "\n"
        
        class_text.insert(tk.END, class_report)
        
        # Save buttons
        button_frame = ttk.Frame(results_window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def save_summary():
            filename = filedialog.asksaveasfilename(
                title="Save Evaluation Summary",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w') as f:
                    f.write(results['comparison_report'])
                messagebox.showinfo("Success", f"Summary saved to: {filename}")
        
        def save_detailed():
            filename = filedialog.asksaveasfilename(
                title="Save Detailed Metrics",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                # Create DataFrame with all metrics
                data = []
                for clf_name in ['SVM', 'KNN', 'LogisticRegression']:
                    if clf_name in results and clf_name != 'training_metadata':
                        data.append({
                            'Classifier': clf_name,
                            'Accuracy': results[clf_name].get('accuracy', 0),
                            'Precision': results[clf_name].get('precision', 0),
                            'Recall': results[clf_name].get('recall', 0),
                            'F1_Score': results[clf_name].get('f1_score', 0),
                            'CV_Accuracy_Mean': results[clf_name].get('cv_accuracy_mean', 'N/A'),
                            'CV_Accuracy_Std': results[clf_name].get('cv_accuracy_std', 'N/A'),
                            'Training_Time': results[clf_name].get('training_time', 0),
                            'Inference_Time': results[clf_name].get('inference_time', 0),
                            'Inference_Speed': results[clf_name].get('inference_speed', 0)
                        })
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Detailed metrics saved to: {filename}")
        
        ttk.Button(button_frame, text="💾 Save Summary", command=save_summary).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💾 Save Detailed CSV", command=save_detailed).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Close", command=results_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def apply_settings(self):
        """Apply threshold settings"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        try:
            self.system.confidence_threshold = self.confidence_var.get()
            self.system.verifier.similarity_threshold = self.verification_var.get()
            
            messagebox.showinfo("Success", "Settings applied successfully!")
            self.refresh_dashboard()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {e}")
    
    def save_system(self):
        """Save system state"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        try:
            self.system.save_system()
            messagebox.showinfo("Success", "System state saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save system: {e}")
    
    def reload_system(self):
        """Reload system"""
        if not self.is_initialized:
            messagebox.showerror("Error", "System not initialized!")
            return
        
        try:
            self.system.load_system()
            messagebox.showinfo("Success", "System reloaded successfully!")
            self.refresh_dashboard()
            self.refresh_users_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload system: {e}")
    
    def refresh_system_info(self):
        """Refresh system information"""
        if not self.is_initialized:
            self.system_info_text.delete(1.0, tk.END)
            self.system_info_text.insert(tk.END, "System not initialized.")
            return
        
        try:
            status = self.system.get_system_status()
            
            info = f"""SYSTEM INFORMATION
{'='*50}

System Configuration:
• System Type: {status['system_type']}
• Enrolled Users: {status['enrolled_users']}
• Total Embeddings: {status['total_embeddings']}
• Active Classifiers: {', '.join(status['active_classifiers'])}
• Confidence Threshold: {status['confidence_threshold']:.3f}
• Verification Threshold: {status['verification_threshold']:.3f}

Hardware Information:
• Processing Device: {self.system.extractor.device}
• Model Path: {self.system.model_path}
• Database Path: {self.system.database.database_path}

Classifier Performance:
"""
            
            perf = status['classifier_performance']
            for clf_name, metrics in perf.items():
                if metrics['total'] > 0:
                    accuracy = (metrics['correct'] / metrics['total']) * 100
                    info += f"• {clf_name}: {accuracy:.1f}% accuracy ({metrics['correct']}/{metrics['total']} correct)\n"
                else:
                    info += f"• {clf_name}: No predictions yet\n"
            
            info += f"\nDatabase Statistics:\n"
            try:
                users = self.system.get_enrolled_users()
                if not users.empty:
                    info += f"• Oldest enrollment: {users['enrollment_date'].min()}\n"
                    info += f"• Latest update: {users['last_updated'].max()}\n"
                    info += f"• Average samples per user: {users['total_embeddings'].mean():.1f}\n"
                
                # Attendance stats
                report = self.system.generate_report()
                if not report.empty:
                    info += f"• Total attendance records: {len(report)}\n"
                    info += f"• First attendance: {report['date'].min()}\n"
                    info += f"• Last attendance: {report['date'].max()}\n"
                else:
                    info += "• No attendance records yet\n"
                    
            except Exception as e:
                info += f"• Error retrieving database stats: {e}\n"
            
            self.system_info_text.delete(1.0, tk.END)
            self.system_info_text.insert(tk.END, info)
            
        except Exception as e:
            self.system_info_text.delete(1.0, tk.END)
            self.system_info_text.insert(tk.END, f"Error retrieving system info: {e}")

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()