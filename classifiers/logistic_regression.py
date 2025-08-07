import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from face_core import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    """Logistic Regression-based face classifier with cosine similarity fallback"""
    def __init__(self, C=1.0, similarity_weight=0.5):
        self.C = C
        self.similarity_weight = similarity_weight
        self.lr_model = None
        self.label_encoder = None
        self.known_embeddings = []
        self.known_names = []
        self.is_single_person = False

    def train(self, embeddings, names):
        if len(embeddings) == 0:
            print("No embeddings provided for training")
            return False
        self.known_embeddings = embeddings.copy()
        self.known_names = names.copy()
        X = np.array(embeddings)
        y = np.array(names)
        unique_classes = len(np.unique(y))
        if unique_classes == 1:
            print(f"Single person mode: {np.unique(y)[0]}")
            print("Using distance-based recognition only")
            self.is_single_person = True
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            return True
        self.is_single_person = False
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.lr_model = LogisticRegression(C=self.C, max_iter=1000, solver='lbfgs', multi_class='auto')
        try:
            self.lr_model.fit(X, y_encoded)
            print(f"Logistic Regression trained successfully with {unique_classes} classes")
            print(f"Classes: {list(self.label_encoder.classes_)}")
            return True
        except Exception as e:
            print(f"Logistic Regression training failed: {e}")
            return False

    def predict(self, embedding):
        if self.label_encoder is None:
            return "Unknown", 0.0
        embedding = np.array(embedding).reshape(1, -1)
        similarities = [cosine_similarity(embedding, np.array(known_emb).reshape(1, -1))[0][0] for known_emb in self.known_embeddings]
        max_similarity = max(similarities) if similarities else 0.0
        best_similarity_idx = np.argmax(similarities) if similarities else 0
        if self.is_single_person:
            predicted_name = self.label_encoder.classes_[0]
            return predicted_name, max_similarity
        if self.lr_model is None:
            return "Unknown", 0.0
        try:
            probabilities = self.lr_model.predict_proba(embedding)[0]
            predicted_class = self.lr_model.predict(embedding)[0]
            lr_confidence = np.max(probabilities)
            predicted_name = self.label_encoder.inverse_transform([predicted_class])[0]
            final_confidence = (1 - self.similarity_weight) * lr_confidence + self.similarity_weight * max_similarity
            return predicted_name, final_confidence
        except Exception as e:
            print(f"Logistic Regression prediction error: {e}")
            if similarities:
                best_name = self.known_names[best_similarity_idx]
                return best_name, max_similarity
            return "Unknown", 0.0

    def save(self, filepath):
        model_data = {
            'lr_model': self.lr_model,
            'label_encoder': self.label_encoder,
            'known_embeddings': self.known_embeddings,
            'known_names': self.known_names,
            'is_single_person': self.is_single_person,
            'C': self.C,
            'similarity_weight': self.similarity_weight
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Logistic Regression classifier saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving Logistic Regression classifier: {e}")
            return False

    def load(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.lr_model = model_data['lr_model']
            self.label_encoder = model_data['label_encoder']
            self.known_embeddings = model_data['known_embeddings']
            self.known_names = model_data['known_names']
            self.is_single_person = model_data['is_single_person']
            self.C = model_data.get('C', 1.0)
            self.similarity_weight = model_data.get('similarity_weight', 0.5)
            print(f"Logistic Regression classifier loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading Logistic Regression classifier: {e}")
            return False

    def get_name(self):
        return "LogisticRegression"
