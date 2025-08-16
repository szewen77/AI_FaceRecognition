# main_knn.py - Face Recognition with KNN Classifier
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add classifiers directory to path
sys.path.append('classifiers')

from face_core import FaceRecognitionSystem
from knn_classifier import KNNClassifier

def batch_enroll_from_folder(system, batch_folder):
    """Enroll multiple people from folder structure"""
    from pathlib import Path
    
    batch_path = Path(batch_folder)
    if not batch_path.exists():
        print(f"Batch folder not found: {batch_folder}")
        return
    
    print(f"Scanning for people in: {batch_folder}")
    
    # Find all subdirectories (each represents a person)
    person_folders = [f for f in batch_path.iterdir() if f.is_dir()]
    
    if not person_folders:
        print("No person folders found!")
        return
    
    print(f"Found {len(person_folders)} people to enroll:")
    for folder in person_folders:
        print(f"  - {folder.name}")
    
    # Enroll each person
    successful_enrollments = 0
    
    for person_folder in person_folders:
        person_name = person_folder.name
        print(f"\n=== Enrolling {person_name} ===")
        
        # Count images in folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in person_folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"  ⚠️  No images found for {person_name}")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Enroll person
        success = system.enroll_person(
            name=person_name,
            images_path=str(person_folder),
            webcam_capture=False,
            num_samples=0
        )
        
        if success:
            print(f"  ✅ Successfully enrolled {person_name}")
            successful_enrollments += 1
        else:
            print(f"  ❌ Failed to enroll {person_name}")
    
    print(f"\n=== Batch Enrollment Complete ===")
    print(f"Successfully enrolled: {successful_enrollments}/{len(person_folders)} people")

def main():
    parser = argparse.ArgumentParser(description='Face Recognition Attendance System with KNN')
    
    # Basic operations
    parser.add_argument('--mode', choices=['enroll', 'run', 'report'], 
                       default='run', help='Operation mode')
    
    # Enrollment options
    parser.add_argument('--name', type=str, help='Name for single person enrollment')
    parser.add_argument('--names', nargs='+', help='Names for multiple people webcam enrollment')
    parser.add_argument('--images', type=str, help='Path to images folder for single person')
    parser.add_argument('--batch-folder', type=str, help='Path to folder containing subfolders for each person')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for enrollment')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples for webcam enrollment')
    
    # System settings
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--model-path', type=str, default='models/', help='Path to save models')
    parser.add_argument('--db-path', type=str, default='attendance.db', help='Database path')
    
    # KNN parameters
    parser.add_argument('--k-neighbors', type=int, default=5, 
                       help='Number of neighbors for KNN (k value)')
    parser.add_argument('--knn-weights', choices=['uniform', 'distance'], default='distance',
                       help='Weight function used in KNN prediction')
    parser.add_argument('--knn-metric', choices=['euclidean', 'manhattan', 'cosine', 'minkowski'], 
                       default='euclidean', help='Distance metric for KNN')
    parser.add_argument('--similarity-weight', type=float, default=0.5,
                       help='Weight for cosine similarity (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Create KNN classifier with parameters
    classifier = KNNClassifier(
        n_neighbors=args.k_neighbors,
        weights=args.knn_weights,
        metric=args.knn_metric,
        similarity_weight=args.similarity_weight
    )
    
    # Initialize system
    system = FaceRecognitionSystem(
        classifier=classifier,
        model_path=args.model_path,
        database_path=args.db_path,
        confidence_threshold=args.threshold
    )
    
    print(f"Initialized Face Recognition System with KNN")
    print(f"KNN Parameters: k={args.k_neighbors}, weights={args.knn_weights}, metric={args.knn_metric}")
    print(f"Similarity weight: {args.similarity_weight}, Confidence threshold: {args.threshold}")
    
    # Execute based on mode
    if args.mode == 'enroll':
        # Batch enrollment from folder structure
        if args.batch_folder:
            batch_enroll_from_folder(system, args.batch_folder)
        
        # Multiple people with webcam
        elif args.names and args.webcam:
            for name in args.names:
                print(f"\n=== Enrolling {name} ===")
                success = system.enroll_person(
                    name=name,
                    webcam_capture=True,
                    num_samples=args.samples
                )
                if success:
                    print(f"✅ Successfully enrolled {name}")
                else:
                    print(f"❌ Failed to enroll {name}")
        
        # Single person enrollment
        elif args.name:
            success = system.enroll_person(
                name=args.name,
                images_path=args.images,
                webcam_capture=args.webcam,
                num_samples=args.samples
            )
            
            if success:
                print(f"✅ Successfully enrolled {args.name}")
            else:
                print(f"❌ Failed to enroll {args.name}")
        
        else:
            print("Error: Please specify enrollment method:")
            print("  --batch-folder <folder>     : Batch enroll from image folders")
            print("  --name <name> --webcam      : Single person via webcam")
            print("  --name <name> --images <dir>: Single person from images")
            print("  --names <name1> <name2> --webcam: Multiple people via webcam")
            return
    
    elif args.mode == 'run':
        print("Starting live attendance system with KNN...")
        print("Press 'q' to quit, 's' to save screenshot")
        system.run_live_attendance()
    
    elif args.mode == 'report':
        df = system.generate_report()
        print("\n=== Attendance Report ===")
        if not df.empty:
            print(df.to_string(index=False))
            
            # Show statistics
            print(f"\nTotal attendance records: {len(df)}")
            print(f"Unique people: {df['name'].nunique()}")
            print(f"Average confidence: {df['confidence'].mean():.3f}")
            
            # Show daily summary
            if 'date' in df.columns:
                daily_summary = df.groupby('date')['name'].count()
                print(f"\nDaily attendance summary:")
                for date, count in daily_summary.items():
                    print(f"  {date}: {count} people")
        else:
            print("No attendance records found.")
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_report_knn_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nReport saved to: {filename}")
        
        # Show enrolled users
        users_df = system.database.get_enrolled_users()
        print(f"\n=== Enrolled Users ({len(users_df)} total) ===")
        if not users_df.empty:
            print(users_df.to_string(index=False))
        else:
            print("No users enrolled yet.")

if __name__ == "__main__":
    main()
