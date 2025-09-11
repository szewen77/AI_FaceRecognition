# kaggle_dataset_integration.py - Kaggle Face Recognition Dataset Integration
import kagglehub
import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import tarfile
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleFaceDatasetIntegrator:
    """
    Integrates Kaggle face recognition datasets into the existing face recognition system
    """
    
    def __init__(self, target_folder: str = "KaggleDatasets", max_people: int = 100, max_images_per_person: int = 20, min_images_per_person: int = 2):
        """
        Initialize the Kaggle dataset integrator
        
        Args:
            target_folder: Main folder to store all Kaggle datasets (organized by dataset name)
            max_people: Maximum number of people to extract from dataset
            max_images_per_person: Maximum images per person to use
            min_images_per_person: Minimum images required per person (people with fewer will be skipped)
        """
        self.main_target_folder = Path(target_folder)
        self.max_people = max_people
        self.max_images_per_person = max_images_per_person
        self.min_images_per_person = min_images_per_person
        
        # Create main target folder if it doesn't exist
        self.main_target_folder.mkdir(exist_ok=True)
        
    def download_and_prepare_dataset(self, dataset_name: str = "vasukipatel/face-recognition-dataset") -> Dict:
        """
        Download and prepare the Kaggle dataset for face recognition system
        
        Args:
            dataset_name: Kaggle dataset identifier
            
        Returns:
            Dictionary with preparation results
        """
        try:
            logger.info(f"🔄 Starting download of Kaggle dataset: {dataset_name}")
            
            # Create a clean dataset-specific folder
            dataset_clean_name = self._clean_dataset_name(dataset_name)
            self.target_folder = self.main_target_folder / dataset_clean_name
            
            logger.info(f"📁 Organizing dataset in: {self.target_folder}")
            
            # Download the dataset
            dataset_path = kagglehub.dataset_download(dataset_name)
            logger.info(f"✅ Dataset downloaded to: {dataset_path}")
            
            # Analyze the dataset structure
            dataset_info = self._analyze_dataset_structure(dataset_path)
            logger.info(f"📊 Dataset analysis: {dataset_info}")
            
            # Organize the dataset for face recognition system
            organized_info = self._organize_dataset_for_enrollment(dataset_path)
            
            return {
                "success": True,
                "dataset_path": dataset_path,
                "organized_path": str(self.target_folder),
                "dataset_info": dataset_info,
                "organized_info": organized_info,
                "download_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to download/prepare dataset: {e}")
            return {
                "success": False,
                "error": str(e),
                "download_time": datetime.now().isoformat()
            }
    
    def _analyze_dataset_structure(self, dataset_path: str) -> Dict:
        """Analyze the structure of the downloaded dataset"""
        dataset_path = Path(dataset_path)
        
        info = {
            "total_files": 0,
            "image_files": 0,
            "directories": 0,
            "file_types": {},
            "structure": "unknown"
        }
        
        try:
            # Count files and analyze structure
            for item in dataset_path.rglob("*"):
                if item.is_file():
                    info["total_files"] += 1
                    
                    # Check if it's an image file
                    if item.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                        info["image_files"] += 1
                    
                    # Count file types
                    ext = item.suffix.lower()
                    info["file_types"][ext] = info["file_types"].get(ext, 0) + 1
                
                elif item.is_dir():
                    info["directories"] += 1
            
            # Determine structure type
            if info["directories"] > 0:
                # Check if it's organized by person folders
                person_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
                if person_folders:
                    info["structure"] = "person_folders"
                    info["potential_people"] = len(person_folders)
                else:
                    info["structure"] = "flat_with_subdirs"
            else:
                info["structure"] = "flat"
                
        except Exception as e:
            logger.error(f"Error analyzing dataset structure: {e}")
            info["error"] = str(e)
            
        return info
    
    def _extract_compressed_files(self, dataset_path: Path) -> Path:
        """Extract compressed files if found in dataset"""
        extracted_path = dataset_path / "extracted"
        
        # Look for compressed files
        for item in dataset_path.iterdir():
            if item.is_file():
                try:
                    if item.suffix.lower() in {'.tgz', '.tar.gz'}:
                        logger.info(f"🔄 Extracting {item.name}...")
                        extracted_path.mkdir(exist_ok=True)
                        with tarfile.open(item, 'r:gz') as tar:
                            tar.extractall(extracted_path)
                        logger.info(f"✅ Extracted {item.name}")
                        return extracted_path
                    
                    elif item.suffix.lower() in {'.zip'}:
                        logger.info(f"🔄 Extracting {item.name}...")
                        extracted_path.mkdir(exist_ok=True)
                        with zipfile.ZipFile(item, 'r') as zip_ref:
                            zip_ref.extractall(extracted_path)
                        logger.info(f"✅ Extracted {item.name}")
                        return extracted_path
                        
                except Exception as e:
                    logger.error(f"Failed to extract {item.name}: {e}")
        
        # Return original path if no extraction needed
        return dataset_path
    
    def _organize_dataset_for_enrollment(self, dataset_path: str) -> Dict:
        """
        Organize the dataset into the format expected by the face recognition system
        (person_name/image1.jpg, image2.jpg, ...)
        """
        dataset_path = Path(dataset_path)
        organized_info = {
            "people_processed": 0,
            "images_processed": 0,
            "skipped_people": 0,
            "errors": []
        }
        
        try:
            # Clear existing organized data
            if self.target_folder.exists():
                shutil.rmtree(self.target_folder)
            self.target_folder.mkdir(exist_ok=True)
            
            # Extract compressed files if needed
            working_path = self._extract_compressed_files(dataset_path)
            
            # Handle different dataset structures
            people_count = 0
            
            # Look for person folders in the dataset
            potential_person_folders = []
            
            # Search for directories that might contain person images
            for item in working_path.rglob("*"):
                if item.is_dir():
                    # Check if this directory contains images
                    image_files = list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + \
                                 list(item.glob("*.png")) + list(item.glob("*.bmp"))
                    
                    if image_files and len(image_files) >= self.min_images_per_person:  # Use configurable minimum
                        potential_person_folders.append((item, image_files))
            
            # If no person folders found, try to organize from flat structure
            if not potential_person_folders:
                potential_person_folders = self._organize_flat_structure(working_path)
            
            # Process each person
            for person_folder, image_files in potential_person_folders[:self.max_people]:
                if people_count >= self.max_people:
                    break
                
                person_name = self._clean_person_name(person_folder.name)
                target_person_folder = self.target_folder / person_name
                target_person_folder.mkdir(exist_ok=True)
                
                # Copy images (limit per person)
                images_copied = 0
                for img_file in image_files[:self.max_images_per_person]:
                    try:
                        target_img_path = target_person_folder / f"{person_name}_{images_copied + 1}{img_file.suffix}"
                        shutil.copy2(img_file, target_img_path)
                        images_copied += 1
                        organized_info["images_processed"] += 1
                        
                    except Exception as e:
                        organized_info["errors"].append(f"Failed to copy {img_file}: {e}")
                
                if images_copied >= self.min_images_per_person:
                    people_count += 1
                    organized_info["people_processed"] += 1
                    logger.info(f"✅ Organized {person_name}: {images_copied} images (min: {self.min_images_per_person})")
                else:
                    organized_info["skipped_people"] += 1
                    logger.info(f"⏭️  Skipped {person_name}: only {images_copied} images (need min: {self.min_images_per_person})")
                    # Remove folder with insufficient images
                    if target_person_folder.exists():
                        shutil.rmtree(target_person_folder)
            
            logger.info(f"🎯 Dataset organization complete: {people_count} people, {organized_info['images_processed']} images")
            
        except Exception as e:
            logger.error(f"Error organizing dataset: {e}")
            organized_info["errors"].append(f"Organization error: {e}")
        
        return organized_info
    
    def _organize_flat_structure(self, dataset_path: Path) -> List:
        """Handle flat dataset structure by grouping similar named files"""
        # This is a fallback for datasets that don't have clear person folders
        # You might need to customize this based on the specific dataset structure
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(dataset_path.rglob(ext))
        
        # Group by filename patterns (this is dataset-specific)
        # For now, create artificial groups
        groups = []
        current_group = []
        
        for i, img_file in enumerate(image_files):
            current_group.append(img_file)
            
            # Create groups of 10-20 images each
            if len(current_group) >= 15 or i == len(image_files) - 1:
                if current_group:
                    # Create a synthetic person folder
                    person_name = f"Person_{len(groups) + 1:03d}"
                    synthetic_folder = dataset_path / person_name
                    groups.append((synthetic_folder, current_group.copy()))
                    current_group = []
        
        return groups
    
    def _clean_dataset_name(self, dataset_name: str) -> str:
        """Clean dataset name to create a filesystem-safe folder name"""
        import re
        # Extract just the dataset part (remove username/)
        if '/' in dataset_name:
            dataset_part = dataset_name.split('/')[-1]
        else:
            dataset_part = dataset_name
        
        # Clean and format
        cleaned = re.sub(r'[^\w\s-]', '', dataset_part)
        cleaned = re.sub(r'[-\s]+', '_', cleaned)
        cleaned = cleaned.strip('_')[:30]  # Limit length
        
        # Make it more readable
        if 'face' in cleaned.lower():
            return f"Face_Recognition_Dataset"
        else:
            return f"Kaggle_{cleaned.title()}"
    
    def _clean_person_name(self, name: str) -> str:
        """Clean person name to be filesystem-safe"""
        # Remove special characters and spaces
        import re
        cleaned = re.sub(r'[^\w\s-]', '', name)
        cleaned = re.sub(r'[-\s]+', '_', cleaned)
        return cleaned.strip('_')[:50]  # Limit length
    
    def get_organized_dataset_info(self) -> Dict:
        """Get information about the organized dataset"""
        if not self.target_folder.exists():
            return {"error": "No organized dataset found"}
        
        info = {
            "total_people": 0,
            "total_images": 0,
            "people_details": []
        }
        
        for person_folder in self.target_folder.iterdir():
            if person_folder.is_dir():
                image_count = len([f for f in person_folder.iterdir() 
                                 if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
                
                info["total_people"] += 1
                info["total_images"] += image_count
                info["people_details"].append({
                    "name": person_folder.name,
                    "image_count": image_count
                })
        
        return info

def main():
    """Example usage of the Kaggle dataset integrator"""
    
    print("🎯 Kaggle Face Recognition Dataset Integrator")
    print("=" * 50)
    
    # Initialize integrator
    integrator = KaggleFaceDatasetIntegrator(
        target_folder="kaggle_faces",
        max_people=50,  # Start with 50 people for testing
        max_images_per_person=15
    )
    
    # Download and prepare dataset
    result = integrator.download_and_prepare_dataset("vasukipatel/face-recognition-dataset")
    
    if result["success"]:
        print("✅ Dataset integration successful!")
        print(f"📁 Organized dataset location: {result['organized_path']}")
        print(f"📊 Dataset info: {result['dataset_info']}")
        print(f"🎯 Organization results: {result['organized_info']}")
        
        # Show organized dataset info
        organized_info = integrator.get_organized_dataset_info()
        print(f"\n📈 Final organized dataset:")
        print(f"   👥 Total people: {organized_info.get('total_people', 0)}")
        print(f"   🖼️  Total images: {organized_info.get('total_images', 0)}")
        
    else:
        print("❌ Dataset integration failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
