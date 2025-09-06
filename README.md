# 🎯 Multi-Classifier Face Recognition Attendance System

A comprehensive face recognition system that uses multiple machine learning algorithms (SVM, KNN, Logistic Regression) with FaceNet embeddings for accurate face identification and attendance tracking.

## 🌟 Features

### **Core Functionality**
- **Multi-Algorithm Approach**: Uses SVM, KNN, and Logistic Regression with majority voting
- **FaceNet Integration**: State-of-the-art face embeddings using pre-trained FaceNet model
- **Real-time Recognition**: Live webcam attendance marking with confidence thresholds
- **Database Management**: SQLite database for user enrollment and attendance records
- **Modern GUI**: User-friendly interface built with tkinter

### **Advanced Features**
- **Stratified Train/Test Split**: Reliable evaluation with 80/20 data split
- **Face Verification**: Prevents duplicate enrollments with similarity checking
- **Performance Tracking**: Real-time monitoring of each classifier's accuracy
- **Comprehensive Evaluation**: Detailed metrics and comparison reports
- **Bulk Enrollment**: Support for dataset folders and individual image uploads

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

1. Go to "Live Attendance" tab
2. Click "🎥 Start Live Attendance"
3. Position faces in front of camera
4. System will automatically recognize and mark attendance
5. Press 'q' to quit, 's' for screenshot, 'p' for performance report

### **Reports & Analysis**

#### **Performance Reports**
- **🎯 Performance Report**: Real-time classifier accuracy tracking
- **📊 Comprehensive Evaluation**: Detailed metrics with train/test split
- **🔄 Retrain & Evaluate**: Retrain all algorithms with current data

#### **Attendance Reports**
- **📋 Generate Report**: View attendance records by date range
- **💾 Export CSV**: Export attendance data to CSV format

## 🔧 Configuration

### **Threshold Settings**
- **Confidence Threshold**: Minimum confidence for recognition (default: 0.7)
- **Verification Threshold**: Minimum similarity for face verification (default: 0.7)

### **Algorithm Parameters**
- **SVM**: Linear kernel with probability estimation
- **KNN**: 3 nearest neighbors with cosine distance
- **Logistic Regression**: L2 regularization with 1000 max iterations

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
- `run_live_attendance()`: Start real-time attendance
- `train_test_evaluate_and_save()`: Train and evaluate with train/test split
- `bulk_enroll_from_directories_and_evaluate()`: Bulk enrollment from folders

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

**Last Updated**: September 2025  
**Version**: 2.0  
**Status**: Active Development
