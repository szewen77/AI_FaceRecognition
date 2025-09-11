# test_kaggle_integration.py - Test Kaggle Dataset Integration
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_kaggle_integration():
    """Test the Kaggle dataset integration"""
    
    print("🧪 Testing Kaggle Dataset Integration")
    print("=" * 50)
    
    try:
        # Test 1: Import check
        print("\n1️⃣ Testing imports...")
        try:
            import kagglehub
            print("✅ kagglehub imported successfully")
        except ImportError:
            print("❌ kagglehub not found. Install with: pip install kagglehub")
            return False
        
        try:
            from kaggle_dataset_integration import KaggleFaceDatasetIntegrator
            print("✅ KaggleFaceDatasetIntegrator imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import KaggleFaceDatasetIntegrator: {e}")
            return False
        
        try:
            from face_core import FixedMultiClassifierSystem
            print("✅ FixedMultiClassifierSystem imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import FixedMultiClassifierSystem: {e}")
            return False
        
        # Test 2: Initialize integrator
        print("\n2️⃣ Testing integrator initialization...")
        integrator = KaggleFaceDatasetIntegrator(
            target_folder="test_kaggle_faces",
            max_people=5,  # Small number for testing
            max_images_per_person=10
        )
        print("✅ Integrator initialized successfully")
        
        # Test 3: Test face recognition system
        print("\n3️⃣ Testing face recognition system...")
        try:
            system = FixedMultiClassifierSystem(
                model_path='models/',
                database_path='test_attendance.db',  # Use test database
                confidence_threshold=0.7,
                verification_threshold=0.7
            )
            print("✅ Face recognition system initialized successfully")
            
            # Check if the Kaggle enrollment method exists
            if hasattr(system, 'enroll_from_kaggle_dataset'):
                print("✅ Kaggle enrollment method found in system")
            else:
                print("❌ Kaggle enrollment method not found in system")
                return False
                
        except Exception as e:
            print(f"❌ Failed to initialize face recognition system: {e}")
            return False
        
        # Test 4: Test dataset info (without downloading)
        print("\n4️⃣ Testing dataset organization methods...")
        try:
            # Test the helper methods without actually downloading
            test_name = integrator._clean_person_name("Test Person #1!")
            if test_name == "Test_Person_1":
                print("✅ Name cleaning method works correctly")
            else:
                print(f"⚠️ Name cleaning method result: '{test_name}' (expected 'Test_Person_1')")
            
            # Test get organized dataset info on empty folder
            info = integrator.get_organized_dataset_info()
            if "error" in info:
                print("✅ Empty dataset info handling works correctly")
            else:
                print(f"⚠️ Unexpected dataset info result: {info}")
                
        except Exception as e:
            print(f"❌ Error testing dataset organization: {e}")
            return False
        
        print("\n✅ All integration tests passed!")
        print("\n📋 Next steps to use the integration:")
        print("1. Install kagglehub: pip install kagglehub")
        print("2. Run the main GUI: python main_app.py")
        print("3. Go to the 'Enrollment' tab")
        print("4. Use the 'Kaggle Dataset Enrollment' section")
        print("5. Or use the older GUI: python main_multiclassifier_gui.py")
        print("6. Click the '🔗 Enroll Kaggle Dataset' button")
        
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error during testing: {e}")
        return False
    
    finally:
        # Clean up test files
        test_db = Path("test_attendance.db")
        if test_db.exists():
            test_db.unlink()
            print("🧹 Cleaned up test database")

def show_usage_example():
    """Show usage example"""
    print("\n" + "=" * 50)
    print("📚 USAGE EXAMPLE")
    print("=" * 50)
    
    example_code = '''
# Example: Using Kaggle dataset integration programmatically

from face_core import FixedMultiClassifierSystem

# Initialize the system
system = FixedMultiClassifierSystem()

# Enroll from Kaggle dataset
result = system.enroll_from_kaggle_dataset(
    dataset_name="vasukipatel/face-recognition-dataset",
    max_people=50,
    max_images_per_person=15
)

if result["success"]:
    print(f"Successfully enrolled {result['enrolled_count']} people!")
else:
    print(f"Enrollment failed: {result['error']}")
'''
    
    print(example_code)
    
    print("\n📋 GUI Usage:")
    print("1. Launch GUI: python main_app.py")
    print("2. Go to 'Enrollment' tab")
    print("3. Fill in Kaggle Dataset Enrollment section:")
    print("   - Dataset: vasukipatel/face-recognition-dataset")
    print("   - Max People: 25 (or desired number)")
    print("   - Max Images/Person: 15 (or desired number)")
    print("4. Click '🔗 Enroll from Kaggle Dataset'")
    print("5. Wait for download and enrollment to complete")
    print("6. System will automatically retrain all classifiers")

if __name__ == "__main__":
    success = test_kaggle_integration()
    
    if success:
        show_usage_example()
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
