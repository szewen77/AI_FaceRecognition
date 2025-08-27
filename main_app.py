import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import time
from datetime import datetime, date
from pathlib import Path
import pandas as pd

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
Live Attendance Instructions:
• Ensure good lighting conditions
• Position camera at eye level
• Multiple people can be recognized simultaneously
• Attendance is automatically marked (once per day per person)
• Debug information will appear in the log below

Camera Controls (when live window is active):
• Press 'q' to quit
• Press 's' to save screenshot
• Press 'p' to print performance report
• Press 'd' to toggle debug mode
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
        folder_path = self.folder_path_var.get().strip()
        
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
        self.folder_path_var.set("")
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
            self.folder_path_var.set(folder_path)

    def browse_files(self):
        file_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
        if file_paths:
            self.files_path_var.set("; ".join(file_paths))
            self.selected_files = list(file_paths)

    def start_file_enrollment(self):
        if not hasattr(self, "selected_files") or not self.selected_files:
            messagebox.showerror("Error", "Please select image files first!")
            return
        # process images in self.selected_files...
        messagebox.showinfo("Enrollment", f"Enrolled {len(self.selected_files)} images")

    
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
                self.log_to_attendance("🎥 Starting live attendance monitoring...")
                self.log_to_attendance("📊 Multi-classifier system active")
                self.log_to_attendance("🎮 Camera controls: 'q'=quit, 's'=screenshot, 'p'=performance, 'd'=debug")
                
                self.system.run_live_attendance()
                
                self.root.after(0, self.on_attendance_stopped)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_attendance_error(str(e)))
        
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
        self.log_to_attendance(f"❌ Attendance error: {error}")
        messagebox.showerror("Attendance Error", f"Attendance error: {error}")
    
    def refresh_users_list(self):
        """Refresh the users list"""
        if not self.is_initialized:
            return
        
        try:
            # Clear existing items
            for item in self.users_tree.get_children():
                self.users_tree.delete(item)
            
            users = self.system.get_enrolled_users()
            
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