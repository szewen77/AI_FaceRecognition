# classifiers/svm_classifier.py - SVM-based face classifier
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from face_core import BaseClassifier

class SVMClassifier(BaseClassifier):
    """SVM-based face classifier with cosine similarity fallback"""
    
    def __init__(self, kernel='linear', C=1.0, probability=True, similarity_weight=0.5):
        """
        Initialize SVM Classifier
        
        Args:
            kernel: SVM kernel type
            C: Regularization parameter
            probability: Enable probability estimates
            similarity_weight: Weight for cosine similarity in final confidence
        """
        self.kernel = kernel
        self.C = C
        self.probability = probability
        self.similarity_weight = similarity_weight
        
        self.svm_model = None
        self.label_encoder = None
        self.known_embeddings = []
        self.known_names = []
        self.is_single_person = False
    
    def train(self, embeddings, names):
        """
        Train SVM classifier
        
        Args:
            embeddings: List of face embeddings
            names: List of corresponding names
            
        Returns:
            bool: Success status
        """
        if len(embeddings) == 0:
            print("No embeddings provided for training")
            return False
        
        # Store embeddings for similarity calculation
        self.known_embeddings = embeddings.copy()
        self.known_names = names.copy()
        
        # Prepare data
        X = np.array(embeddings)
        y = np.array(names)
        
        # Check number of unique classes
        unique_classes = len(np.unique(y))
        
        if unique_classes == 1:
            print(f"Single person mode: {np.unique(y)[0]}")
            print("Using distance-based recognition only")
            self.is_single_person = True
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            return True
        
        # Multiple people - train SVM
        self.is_single_person = False
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train SVM
        self.svm_model = SVC(
            kernel=self.kernel,
            probability=self.probability,
            C=self.C,
            random_state=42
        )
        
        try:
            self.svm_model.fit(X, y_encoded)
            
            # Verify the model was trained properly
            if hasattr(self.svm_model, 'support_vectors_') and self.svm_model.support_vectors_ is not None:
                print(f"SVM trained successfully with {unique_classes} classes")
                print(f"Classes: {list(self.label_encoder.classes_)}")
                print(f"Support vectors: {len(self.svm_model.support_vectors_)}")
                return True
            else:
                print("SVM training completed but model may not be properly fitted")
                # Try with different parameters
                self.svm_model = SVC(
                    kernel='rbf',  # Try RBF kernel instead
                    probability=self.probability,
                    C=self.C,
                    random_state=42,
                    gamma='scale'  # Add gamma parameter
                )
                self.svm_model.fit(X, y_encoded)
                print(f"SVM retrained with RBF kernel - {unique_classes} classes")
                return True
                
        except Exception as e:
            print(f"SVM training failed: {e}")
            # Fallback to linear kernel with different C value
            try:
                self.svm_model = SVC(
                    kernel='linear',
                    probability=self.probability,
                    C=0.1,  # Lower C value
                    random_state=42
                )
                self.svm_model.fit(X, y_encoded)
                print(f"SVM trained with fallback parameters - {unique_classes} classes")
                return True
            except Exception as e2:
                print(f"SVM fallback training also failed: {e2}")
                return False
    
    def predict(self, embedding):
        """
        Predict identity from embedding
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            tuple: (predicted_name, confidence)
        """
        if self.label_encoder is None:
            return "Unknown", 0.0
        
        # Reshape for prediction
        embedding = embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = []
        for known_embedding in self.known_embeddings:
            similarity = cosine_similarity(
                embedding, 
                np.array(known_embedding).reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        max_similarity = max(similarities) if similarities else 0.0
        best_similarity_idx = np.argmax(similarities) if similarities else 0
        
        # Single person mode
        if self.is_single_person:
            predicted_name = self.label_encoder.classes_[0]
            return predicted_name, max_similarity
        
        # Multi-person mode with SVM
        if self.svm_model is None:
            return "Unknown", 0.0
        
        try:
            # Check if SVM model is properly trained
            if not hasattr(self.svm_model, 'support_vectors_') or self.svm_model.support_vectors_ is None:
                print("SVM model not properly trained, using similarity-based prediction")
                if similarities:
                    best_name = self.known_names[best_similarity_idx]
                    return best_name, max_similarity
                return "Unknown", 0.0
            
            # Get SVM prediction
            probabilities = self.svm_model.predict_proba(embedding)[0]
            predicted_class = self.svm_model.predict(embedding)[0]
            svm_confidence = np.max(probabilities)
            
            predicted_name = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Combine SVM confidence and cosine similarity
            final_confidence = (
                (1 - self.similarity_weight) * svm_confidence + 
                self.similarity_weight * max_similarity
            )
            
            return predicted_name, final_confidence
            
        except Exception as e:
            print(f"SVM prediction error: {e}")
            # Fallback to similarity-based prediction
            if similarities:
                best_name = self.known_names[best_similarity_idx]
                return best_name, max_similarity
            return "Unknown", 0.0
    
    def save(self, filepath):
        """Save trained classifier"""
        model_data = {
            'svm_model': self.svm_model,
            'label_encoder': self.label_encoder,
            'known_embeddings': self.known_embeddings,
            'known_names': self.known_names,
            'is_single_person': self.is_single_person,
            'kernel': self.kernel,
            'C': self.C,
            'probability': self.probability,
            'similarity_weight': self.similarity_weight
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"SVM classifier saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving SVM classifier: {e}")
            return False
    
    def load(self, filepath):
        """Load trained classifier"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.svm_model = model_data['svm_model']
            self.label_encoder = model_data['label_encoder']
            self.known_embeddings = model_data['known_embeddings']
            self.known_names = model_data['known_names']
            self.is_single_person = model_data['is_single_person']
            self.kernel = model_data.get('kernel', 'linear')
            self.C = model_data.get('C', 1.0)
            self.probability = model_data.get('probability', True)
            self.similarity_weight = model_data.get('similarity_weight', 0.5)
            
            print(f"SVM classifier loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading SVM classifier: {e}")
            return False
    
    def get_name(self):
        """Return classifier name"""
        return "SVM"
    
    def get_parameters(self):
        """Get current parameters"""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'probability': self.probability,
            'similarity_weight': self.similarity_weight,
            'is_single_person': self.is_single_person,
            'num_classes': len(set(self.known_names)) if self.known_names else 0,
            'total_samples': len(self.known_embeddings)
        }
    
    def set_similarity_weight(self, weight):
        """Adjust the weight of cosine similarity in final prediction"""
        self.similarity_weight = max(0.0, min(1.0, weight))
        print(f"Similarity weight set to {self.similarity_weight}")