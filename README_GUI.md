# Face Recognition Attendance System - GUI Version

A user-friendly graphical interface for the face recognition attendance system with support for multiple classifiers (SVM, KNN, and Logistic Regression).

## 🖥️ GUI Features

### **Classifier Selection**
- **KNN (K-Nearest Neighbors)**: Fast training, good for small datasets, instance-based learning
- **SVM (Support Vector Machine)**: Good generalization, effective in high dimensions, memory efficient  
- **Logistic Regression**: Probabilistic output, linear decision boundary, simple and interpretable

### **Real-time Controls**
- **Start/Stop Camera**: Toggle live face recognition
- **Enroll Students**: Batch enrollment from image folders
- **Generate Report**: Create attendance reports with statistics
- **Compare Classifiers**: Test all classifiers on the same data

### **Live Status Display**
- Current classifier in use
- Number of enrolled people
- Confidence threshold
- Real-time system logs

## 🚀 How to Use

### 1. Launch the GUI
```bash
python main_multiclassifier_gui.py
```

### 2. System Overview
The GUI provides an intuitive interface with:
- **Left Panel**: Classifier selection and control buttons
- **Right Panel**: Live system logs and status updates

### 3. Basic Workflow

#### **Step 1: Select Classifier**
Choose from three classification algorithms:
- Click on **KNN** for fast, instance-based recognition
- Click on **SVM** for robust, generalizable classification  
- Click on **Logistic Regression** for probabilistic predictions

#### **Step 2: Enroll Students**
- Click **"👤 Enroll Students"** to load faces from the `imageFolder`
- The system will process all student images automatically
- Watch the logs for enrollment progress

#### **Step 3: Start Attendance**
- Click **"▶ Start Camera"** to begin live face recognition
- Point the camera at students to identify them
- Attendance will be automatically marked when confidence is high enough

#### **Step 4: View Results**
- Click **"📊 Generate Report"** to see attendance statistics
- Click **"⚖ Compare Classifiers"** to test different algorithms

## 🎯 GUI Components

### **Classifier Selection Panel**
```
🔘 KNN - K-Nearest Neighbors
   • Fast training
   • Good for small datasets  
   • Instance-based learning

🔘 SVM - Support Vector Machine
   • Good generalization
   • Effective in high dimensions
   • Memory efficient

🔘 LogisticRegression - Logistic Regression
   • Probabilistic output
   • Linear decision boundary
   • Simple and interpretable
```

### **Control Buttons**
- **▶ Start Camera** / **⏹ Stop Camera**: Toggle live recognition
- **👤 Enroll Students**: Batch process student images
- **📊 Generate Report**: Create attendance CSV report
- **⚖ Compare Classifiers**: Test all algorithms

### **System Status**
- **Status**: Shows if system is ready and which classifier is active
- **Enrolled**: Number of people in the system
- **Confidence**: Current recognition threshold

### **Live Logs**
Real-time system messages showing:
- ✅ Successful operations
- ❌ Errors and failures  
- ⚠️ Warnings and low confidence detections
- ℹ️ General information and status updates

## 🔧 Technical Details

### **Threading Architecture**
- **Main Thread**: GUI interface and user interactions
- **Camera Thread**: Live video processing and face recognition
- **Log Queue**: Thread-safe message passing between threads

### **Classifier Switching**
- Dynamic classifier switching without restarting
- Automatic model loading/saving
- Preserves enrolled student data across switches

### **Performance Optimization**
- Processes every 5th video frame for smooth performance
- Efficient memory management
- Real-time confidence scoring

## 📊 Demo Features

The GUI is designed for easy demonstration:

1. **Visual Feedback**: Color-coded face detection boxes (green=recognized, red=unknown)
2. **Live Statistics**: Real-time display of system metrics
3. **Classifier Comparison**: Side-by-side testing of all algorithms
4. **Attendance Tracking**: Automatic logging with timestamps and confidence scores

## 🎓 For Students: jianquan, shaorong, szewen

The system comes pre-configured for your three students:
- Each student has their own image folder in `imageFolder/`
- All classifiers can recognize the same enrolled students
- Attendance records track which classifier was used for each detection

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install tkinter opencv-python scikit-learn torch facenet-pytorch pandas
```

### Quick Start
```bash
# 1. Launch GUI
python main_multiclassifier_gui.py

# 2. Click "Enroll Students" to load student data
# 3. Select your preferred classifier (KNN recommended)
# 4. Click "Start Camera" to begin attendance tracking
```

## 📈 Comparison Results

The GUI allows you to compare all three classifiers:

| Feature | KNN | SVM | Logistic Regression |
|---------|-----|-----|-------------------|
| **Training Speed** | Instant | Moderate | Fast |
| **Memory Usage** | High | Low | Low |
| **Interpretability** | High | Medium | High |
| **Probability Output** | Distance-based | Yes | Yes |
| **Non-linear Boundaries** | Yes | Yes (RBF) | No |

## 🎬 Demo Script

For presentations, follow this demo flow:

1. **"Let me show you our face recognition system..."**
   - Launch GUI, explain the three classifiers

2. **"First, we'll enroll our students..."**
   - Click Enroll Students, show progress in logs

3. **"Now let's try different classification algorithms..."**
   - Switch between KNN, SVM, and Logistic Regression
   - Show how the system adapts instantly

4. **"Time for live face recognition!"**
   - Start camera, demonstrate face detection
   - Show attendance marking in real-time

5. **"Let's see the results..."**
   - Generate report showing attendance records
   - Compare classifier performance

## 🐛 Troubleshooting

### Common Issues
- **Camera not starting**: Check if camera is already in use by another application
- **No faces detected**: Ensure good lighting and position face clearly in frame
- **Low accuracy**: Try different classifiers or adjust confidence threshold
- **GUI freezing**: Close and restart the application

### Performance Tips
- Use good lighting for better face detection
- Position camera 2-3 feet from subjects
- Ensure clear, unobstructed view of faces
- KNN typically works best for small student groups

## 🔮 Future Enhancements

- [ ] Real-time confidence threshold adjustment
- [ ] Live camera feed display in GUI
- [ ] Export reports to different formats
- [ ] Student photo management interface
- [ ] Batch attendance reports by date range
- [ ] Integration with school management systems
