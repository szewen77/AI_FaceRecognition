#!/usr/bin/env python3
# launch_gui.py - Simple launcher for the Face Recognition GUI

import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required packages are available"""
    required_packages = [
        'tkinter', 'cv2', 'torch', 'facenet_pytorch', 
        'sklearn', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'tkinter':
                import tkinter
            elif package == 'facenet_pytorch':
                import facenet_pytorch
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Install missing packages with:")
        print("   pip install opencv-python torch facenet-pytorch scikit-learn pandas numpy")
        return False
    
    return True

def check_data():
    """Check if student data is available"""
    image_folder = Path("imageFolder")
    if not image_folder.exists():
        print("⚠️  Warning: imageFolder not found")
        print("   Student enrollment may not work without image data")
        return False
    
    # Count student folders
    student_folders = [f for f in image_folder.iterdir() if f.is_dir()]
    if len(student_folders) == 0:
        print("⚠️  Warning: No student folders found in imageFolder")
        return False
    
    print(f"✅ Found {len(student_folders)} student folders:")
    for folder in student_folders:
        image_count = len([f for f in folder.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        print(f"   - {folder.name}: {image_count} images")
    
    return True

def main():
    print("🎓 Face Recognition Attendance System - GUI Launcher")
    print("=" * 55)
    
    # Check requirements
    print("\n🔍 Checking system requirements...")
    if not check_requirements():
        print("\n💥 Cannot start due to missing packages!")
        return 1
    
    print("✅ All required packages found")
    
    # Check data
    print("\n📁 Checking student data...")
    data_ok = check_data()
    
    # Check models directory
    models_dir = Path("models")
    if not models_dir.exists():
        print("\n📂 Creating models directory...")
        models_dir.mkdir()
    
    # Check classifiers directory
    classifiers_dir = Path("classifiers")
    if not classifiers_dir.exists():
        print("❌ Classifiers directory not found!")
        print("   Make sure you're running from the correct directory")
        return 1
    
    print("\n🚀 Launching GUI application...")
    print("📝 Instructions:")
    print("   1. Select a classifier (KNN recommended for demo)")
    print("   2. Click 'Enroll Students' to load student data")
    print("   3. Click 'Start Camera' to begin face recognition")
    print("   4. Click 'Generate Report' to see attendance results")
    print("\n" + "=" * 55)
    
    # Launch the GUI
    try:
        from main_multiclassifier_gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Failed to import GUI application: {e}")
        print("   Make sure main_multiclassifier_gui.py is in the current directory")
        return 1
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        input("\nPress Enter to exit...")
    sys.exit(exit_code)
