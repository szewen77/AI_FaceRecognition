# 🎯 Multi-Classifier Face Recognition Attendance System

A comprehensive face recognition system that uses multiple machine learning algorithms (SVM, KNN, Logistic Regression) with FaceNet embeddings for accurate face identification and attendance tracking. Features user-friendly live attendance with popup confirmations and comprehensive dataset management.

## 🌟 Features

### **Core Functionality**
- **Multi-Algorithm Approach**: Uses SVM, KNN, and Logistic Regression with majority voting
- **FaceNet Integration**: State-of-the-art face embeddings using pre-trained FaceNet model
- **User-Friendly Live Attendance**: Clean camera interface with automatic popup confirmations
- **Database Management**: SQLite database for user enrollment and attendance records
- **Modern GUI**: Intuitive interface built with tkinter and organized tabs

### **Advanced Features**
- **Stratified Train/Test Split**: Reliable evaluation with 80/20 data split
- **Face Verification**: Prevents duplicate enrollments with similarity checking
- **Performance Tracking**: Real-time monitoring of each classifier's accuracy
- **Comprehensive Evaluation**: Detailed metrics and comparison reports with bar charts
- **Bulk Enrollment**: Support for dataset folders, LFW dataset integration, and individual uploads
- **Smart Dataset Management**: Automatic cleanup of low-sample users and memory optimization
- **Duplicate Prevention**: Intelligent checking to avoid re-enrolling existing users

## 📊 Performance Metrics

Based on recent evaluation results:
- **SVM**: 86.5% accuracy, 89.2% precision, 87.6% F1-score
- **KNN**: 86.5% accuracy, 89.2% precision, 87.6% F1-score  
- **Logistic Regression**: 83.8% accuracy, 86.5% precision, 84.9% F1-score
- **Inference Speed**: 11-57 samples/second depending on algorithm

## 🚀 Quick Start

### **1. Installation**

```bash
# Clone the repository
git clone <repository-url>
cd AI_FaceRecognition

# Install dependencies
pip install -r requirements.txt
```

### **2. Launch the Application**

```bash
# Option 1: Use the batch file (Windows)
launch_gui.bat

# Option 2: Run directly
python main_app.py
```

### **3. First-Time Setup**

1. **Enroll Users**: Go to the "Enrollment" tab
2. **Add Images**: Upload images or use webcam capture
3. **Train System**: Click "🔄 Retrain & Evaluate" to train all algorithms
4. **Start Attendance**: Use the "Live Attendance" tab for real-time recognition

## 📁 Project Structure

```
AI_FaceRecognition/
├── main_app.py                 # Main GUI application
├── face_core.py               # Core face recognition logic
├── classifiers/               # Individual classifier implementations
│   ├── svm_classifier.py
│   ├── knn_classifier.py
│   └── logistic_regression.py
├── models/                    # Trained models and evaluation results
│   ├── *_model.pkl           # Saved classifier models
│   ├── *_results.json        # Evaluation metrics
│   └── algorithm_comparison_*.csv
├── Faces/Faces/              # Face dataset (flat structure)
├── Original Images/          # Original dataset (folder structure)
├── attendance.db             # SQLite database
└── requirements.txt          # Python dependencies
```

## 🎮 Usage Guide

### **Enrollment Process**

#### **Method 1: Webcam Capture**
1. Go to "Enrollment" tab
2. Enter person's name
3. Click "📷 Capture from Webcam"
4. Follow on-screen instructions (press SPACE to capture)

#### **Method 2: Image Upload**
1. Go to "Enrollment" tab
2. Enter person's name
3. Click "📁 Select Images" and choose multiple images
4. Click "✅ Enroll Person"

#### **Method 3: Bulk Enrollment**
```python
from face_core import FixedMultiClassifierSystem

system = FixedMultiClassifierSystem()
# For folder structure: person_name/image1.jpg, image2.jpg...
results = system.enroll_from_single_dataset_root_and_evaluate(
    dataset_root='dataset/', 
    test_size=0.2
)
```

### **Live Attendance**

1. Go to "🎥 Live Attendance" tab
2. Click "▶️ Start Live Attendance"
3. Position faces in front of camera
4. System automatically recognizes and marks attendance
5. **Success popup appears** with confirmation details
6. **Camera window stays open** for continuous monitoring
7. Press 'q' to quit manually when done

**Features:**
- Clean, non-flickering interface
- Automatic popup confirmation when attendance is marked
- 3-second cooldown prevents duplicate markings
- Continuous monitoring for multiple people
- Natural camera window size (no forced resizing)

### **Reports & Analysis**

#### **Performance Reports**
- **🎯 Performance Report**: Real-time classifier accuracy tracking
- **📊 Comprehensive Evaluation**: Detailed metrics with train/test split
- **🔄 Retrain Current Students**: Retrain all algorithms with current enrolled users
- **📊 Generate Performance Charts**: Interactive bar charts showing algorithm performance
- **📊 System Status**: Detailed system information and dataset statistics

#### **Attendance Reports**
- **📋 Generate Report**: View attendance records by date range
- **💾 Export CSV**: Export attendance data to CSV format

### **User Management**

#### **User List & Management**
- **👥 Users Tab**: View all enrolled users with sample counts
- **🧹 Clean Low Samples**: Remove users with fewer than 10 samples
- **👥 Add 100 More Students**: Bulk add students with 10 samples each
- **Total User Count**: Real-time display of enrolled user count
- **Duplicate Prevention**: Automatic checking to avoid re-enrolling existing users

## 🔧 Configuration

### **Threshold Settings**
- **Confidence Threshold**: Minimum confidence for recognition (default: 0.7)
- **Verification Threshold**: Minimum similarity for face verification (default: 0.7)

### **Algorithm Parameters**
- **SVM**: Linear kernel with probability estimation and fallback training
- **KNN**: 5 nearest neighbors with distance weighting and euclidean metric
- **Logistic Regression**: L2 regularization with 1000 max iterations

### **Memory Optimization**
- **Image Limiting**: Maximum 10 images per person for training efficiency
- **Random Sampling**: Balanced dataset with random image selection
- **Large Dataset Handling**: Automatic optimization for datasets with many images per person

## 📈 Evaluation & Metrics

### **Train/Test Split Logic**
- **80% Training**: Used to train all algorithms
- **20% Testing**: Held-out data for unbiased evaluation
- **Stratified Split**: Maintains equal proportion for each person

### **Evaluation Metrics**
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Inference Speed**: Samples processed per second

### **Saved Results**
All evaluation results are automatically saved to `models/`:
- `KNN_results.json`, `SVM_results.json`, `LogisticRegression_results.json`
- `algorithm_comparison_YYYYMMDD_HHMMSS.csv`
- `comprehensive_evaluation_results.json`
- `multi_classifier_system_data.pkl` (system state)

### **Performance Charts**
- **Interactive Bar Charts**: Visual comparison of algorithm performance
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, Training Time, Inference Speed
- **Export Capability**: Save charts as images for reports
- **Fallback Display**: Text-based summary if charts fail to load

## 🛠️ Technical Details

### **Face Recognition Pipeline**
1. **Face Detection**: MTCNN for robust face detection
2. **Face Alignment**: Automatic face cropping and normalization
3. **Feature Extraction**: FaceNet embeddings (512-dimensional vectors)
4. **Classification**: Multi-algorithm approach with majority voting
5. **Verification**: Cosine similarity for enrollment validation

### **Database Schema**
- **enrolled_users**: User information and enrollment dates
- **user_embeddings**: Face embeddings for each user
- **attendance**: Attendance records with timestamps and confidence scores

### **System Requirements**
- **Python**: 3.7+
- **RAM**: 4GB+ recommended
- **GPU**: Optional (CUDA support for faster processing)
- **Camera**: USB webcam for live attendance

## 🔍 Troubleshooting

### **Common Issues**

#### **"No face detected"**
- Ensure good lighting
- Position face clearly in camera view
- Check if face is too small or too large

#### **"System not initialized"**
- Wait for system startup to complete
- Check if all dependencies are installed
- Restart the application

#### **Low recognition accuracy**
- Enroll more images per person (recommended: 10-20)
- Ensure good image quality and variety
- Adjust confidence threshold in settings

#### **Import errors**
```bash
pip install --upgrade -r requirements.txt
```

### **Performance Optimization**
- Use GPU if available (automatic detection)
- Reduce image resolution for faster processing
- Limit number of enrolled users for real-time performance

## 📚 API Reference

### **Core Classes**

#### **FixedMultiClassifierSystem**
Main system class that orchestrates all components.

```python
system = FixedMultiClassifierSystem(
    model_path='models/',
    database_path='attendance.db',
    confidence_threshold=0.7,
    verification_threshold=0.7
)
```

#### **Key Methods**
- `enroll_person()`: Enroll new person with images
- `run_live_attendance()`: Start user-friendly real-time attendance with popup confirmations
- `train_test_evaluate_and_save()`: Train and evaluate with train/test split
- `add_more_students()`: Add additional students with duplicate checking
- `remove_students_with_insufficient_samples()`: Clean up users with low sample counts
- `get_system_status()`: Get detailed system information and statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FaceNet**: Face recognition model by Google
- **MTCNN**: Multi-task CNN for face detection
- **scikit-learn**: Machine learning algorithms
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the evaluation results in `models/`
3. Create an issue with detailed error information

---

## 🆕 Recent Updates

### **Version 2.1 Features**
- ✅ **User-Friendly Live Attendance**: Clean interface with popup confirmations
- ✅ **Smart Dataset Management**: Automatic cleanup and memory optimization
- ✅ **Bulk Student Addition**: Add 100+ students with duplicate prevention
- ✅ **Performance Charts**: Interactive bar charts with fallback display
- ✅ **Enhanced GUI**: Better button placement and user experience
- ✅ **Memory Optimization**: Handles large datasets efficiently
- ✅ **Robust Error Handling**: Improved stability and error recovery

### **Key Improvements**
- **Live Attendance**: Camera window stays open with natural size, popup confirmations
- **User Management**: Total user count display, low-sample cleanup
- **Performance Visualization**: Bar charts with proper y-axis scaling (0-100%)
- **Dataset Integration**: LFW dataset support with configurable limits
- **Code Cleanup**: Removed unused functions, improved organization

---

**Last Updated**: January 2025  
**Version**: 2.1  
**Status**: Active Development
