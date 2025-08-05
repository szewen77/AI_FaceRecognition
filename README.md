# Face Recognition Attendance System

A real-time face recognition attendance system built with PyTorch, FaceNet, and OpenCV.

## Features

- **Real-time face detection** using MTCNN
- **Face recognition** using FaceNet embeddings
- **Attendance tracking** with SQLite database
- **Webcam enrollment** for new users
- **Image-based enrollment** from folders
- **Attendance reports** generation
- **Multi-person support** with SVM classifier

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- scikit-learn
- facenet-pytorch
- pandas
- numpy

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd face-recognition-attendance
   ```

2. **Create virtual environment:**
   ```bash
   conda create -n facenet_env python=3.7
   conda activate facenet_env
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision
   pip install opencv-python
   pip install facenet-pytorch
   pip install scikit-learn
   pip install pandas numpy
   ```

## Usage

### 1. Enroll Users

**Using webcam:**
```bash
python face_attendance.py --mode enroll --name "John" --webcam --samples 5
```

**Using images:**
```bash
python face_attendance.py --mode enroll --name "John" "Jane" --images "imageFolder"
```

**Folder structure for images:**
```
imageFolder/
├── John/
│   ├── john1.jpg
│   └── john2.jpg
└── Jane/
    ├── jane1.jpg
    └── jane2.jpg
```

### 2. Run Attendance System

```bash
python face_attendance.py --mode run
```

### 3. Generate Reports

```bash
python face_attendance.py --mode report
```

## Command Line Arguments

- `--mode`: Operation mode (`enroll`, `run`, `report`)
- `--name`: Name(s) of person(s) to enroll
- `--images`: Path to images folder for enrollment
- `--webcam`: Use webcam for enrollment
- `--samples`: Number of samples for webcam enrollment (default: 10)
- `--threshold`: Confidence threshold (default: 0.7)

## System Architecture

### Components

1. **Face Detection**: MTCNN for real-time face detection
2. **Face Recognition**: FaceNet for 512-dimensional embeddings
3. **Classification**: SVM for multi-person recognition
4. **Database**: SQLite for attendance records
5. **UI**: OpenCV for webcam interface

### Workflow

1. **Enrollment**: Capture face embeddings and train classifier
2. **Detection**: Real-time face detection in webcam feed
3. **Recognition**: Compare embeddings with enrolled users
4. **Attendance**: Mark attendance in database
5. **Reporting**: Generate attendance reports

## File Structure

```
├── face_attendance.py      # Main application
├── requirements.txt        # Dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── models/                # Saved models (auto-created)
├── imageFolder/           # Training images (user-created)
└── attendance.db          # Database (auto-created)
```

## Configuration

### Confidence Threshold
- Default: 0.7
- Higher values = stricter recognition
- Lower values = more permissive

### Model Settings
- Face detection: MTCNN with 160x160 input
- Face recognition: FaceNet with 512-dimensional embeddings
- Classifier: SVM with linear kernel

## Troubleshooting

### Common Issues

1. **No faces detected**: Ensure good lighting and clear faces
2. **Low recognition accuracy**: Increase training samples per person
3. **Model not created**: Need at least 2 embeddings to train classifier
4. **Webcam not working**: Check camera permissions and availability

### Performance Tips

- Use GPU if available for faster processing
- Reduce webcam resolution for better performance
- Process every 5th frame for real-time operation
- Use clear, well-lit images for enrollment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- FaceNet implementation: [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- MTCNN: [MTCNN](https://github.com/ipazc/mtcnn)
- OpenCV for computer vision operations 