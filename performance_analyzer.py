# performance_analyzer.py - Comprehensive Model Performance Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import time
import sys
sys.path.append('classifiers')

from face_core import FaceRecognitionSystem
from svm_classifier import SVMClassifier

class PerformanceAnalyzer:
    """Analyze face recognition model performance"""
    
    def __init__(self, system):
        self.system = system
        self.results = {}
    
    def cross_validation_test(self, cv_folds=5):
        """Perform cross-validation on enrolled data"""
        print(f"\n=== Cross-Validation Analysis ({cv_folds}-fold) ===")
        
        if len(self.system.known_embeddings) < cv_folds:
            print(f"Not enough samples for {cv_folds}-fold CV. Need at least {cv_folds} samples.")
            return None
        
        X = np.array(self.system.known_embeddings)
        y = np.array(self.system.known_names)
        
        # Create a fresh classifier for CV
        cv_classifier = SVMClassifier(
            kernel=self.system.classifier.kernel,
            C=self.system.classifier.C,
            similarity_weight=self.system.classifier.similarity_weight
        )
        
        # Use the underlying SVM model for sklearn cross_val_score
        if hasattr(self.system.classifier, 'svm_model') and self.system.classifier.svm_model:
            scores = cross_val_score(self.system.classifier.svm_model, X, 
                                   self.system.classifier.label_encoder.transform(y), 
                                   cv=cv_folds, scoring='accuracy')
            
            print(f"Cross-validation scores: {scores}")
            print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            
            self.results['cv_scores'] = scores
            self.results['cv_mean'] = scores.mean()
            self.results['cv_std'] = scores.std()
            
            return scores
        else:
            print("SVM model not available for cross-validation")
            return None
    
    def confidence_analysis(self):
        """Analyze confidence scores for all enrolled faces"""
        print(f"\n=== Confidence Score Analysis ===")
        
        confidences = {}
        processing_times = []
        
        for i, (embedding, true_name) in enumerate(zip(self.system.known_embeddings, self.system.known_names)):
            # Measure prediction time
            start_time = time.time()
            predicted_name, confidence = self.system.predict_person(embedding)
            prediction_time = time.time() - start_time
            processing_times.append(prediction_time)
            
            # Store confidence by person
            if true_name not in confidences:
                confidences[true_name] = []
            confidences[true_name].append({
                'confidence': confidence,
                'correct': predicted_name == true_name,
                'predicted': predicted_name,
                'sample_id': i,
                'processing_time': prediction_time
            })
        
        # Analyze results
        print("Per-person confidence analysis:")
        all_confidences = []
        correct_predictions = 0
        total_predictions = 0
        
        for person, person_confidences in confidences.items():
            person_conf_values = [item['confidence'] for item in person_confidences]
            person_accuracy = sum([item['correct'] for item in person_confidences]) / len(person_confidences)
            
            print(f"\n{person}:")
            print(f"  Samples: {len(person_confidences)}")
            print(f"  Accuracy: {person_accuracy:.3f}")
            print(f"  Confidence - Mean: {np.mean(person_conf_values):.3f}, "
                  f"Std: {np.std(person_conf_values):.3f}")
            print(f"  Confidence - Min: {np.min(person_conf_values):.3f}, "
                  f"Max: {np.max(person_conf_values):.3f}")
            
            all_confidences.extend(person_conf_values)
            correct_predictions += sum([item['correct'] for item in person_confidences])
            total_predictions += len(person_confidences)
        
        # Overall statistics
        overall_accuracy = correct_predictions / total_predictions
        avg_processing_time = np.mean(processing_times)
        
        print(f"\n=== Overall Performance ===")
        print(f"Overall accuracy: {overall_accuracy:.3f}")
        print(f"Average confidence: {np.mean(all_confidences):.3f}")
        print(f"Confidence std: {np.std(all_confidences):.3f}")
        print(f"Average processing time: {avg_processing_time*1000:.1f}ms per face")
        
        self.results['confidences'] = confidences
        self.results['overall_accuracy'] = overall_accuracy
        self.results['avg_confidence'] = np.mean(all_confidences)
        self.results['avg_processing_time'] = avg_processing_time
        
        return confidences
    
    def confusion_matrix_analysis(self):
        """Generate confusion matrix for enrolled data"""
        print(f"\n=== Confusion Matrix Analysis ===")
        
        true_labels = []
        predicted_labels = []
        
        for embedding, true_name in zip(self.system.known_embeddings, self.system.known_names):
            predicted_name, confidence = self.system.predict_person(embedding)
            
            true_labels.append(true_name)
            predicted_labels.append(predicted_name)
        
        # Get unique labels
        unique_labels = sorted(list(set(true_labels)))
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
        
        print("Confusion Matrix:")
        print(f"Labels: {unique_labels}")
        print(cm)
        
        # Classification report
        print("\nDetailed Classification Report:")
        report = classification_report(true_labels, predicted_labels, 
                                     target_names=unique_labels, 
                                     output_dict=True)
        print(classification_report(true_labels, predicted_labels, target_names=unique_labels))
        
        self.results['confusion_matrix'] = cm
        self.results['classification_report'] = report
        self.results['unique_labels'] = unique_labels
        
        return cm, report
    
    def threshold_analysis(self, thresholds=None):
        """Analyze performance at different confidence thresholds"""
        print(f"\n=== Threshold Analysis ===")
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        threshold_results = []
        
        for threshold in thresholds:
            correct = 0
            total = 0
            accepted = 0  # Predictions above threshold
            
            for embedding, true_name in zip(self.system.known_embeddings, self.system.known_names):
                predicted_name, confidence = self.system.predict_person(embedding)
                
                total += 1
                if confidence >= threshold:
                    accepted += 1
                    if predicted_name == true_name:
                        correct += 1
            
            accuracy = correct / accepted if accepted > 0 else 0
            acceptance_rate = accepted / total
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'acceptance_rate': acceptance_rate,
                'correct': correct,
                'accepted': accepted,
                'total': total
            })
            
            print(f"Threshold {threshold:.1f}: "
                  f"Accuracy {accuracy:.3f}, "
                  f"Acceptance Rate {acceptance_rate:.3f} "
                  f"({accepted}/{total})")
        
        self.results['threshold_analysis'] = threshold_results
        return threshold_results
    
    def performance_benchmark(self, num_iterations=100):
        """Benchmark processing speed"""
        print(f"\n=== Performance Benchmark ({num_iterations} iterations) ===")
        
        if not self.system.known_embeddings:
            print("No enrolled embeddings for benchmarking")
            return None
        
        # Use first embedding for benchmarking
        test_embedding = self.system.known_embeddings[0]
        
        # Warm up
        for _ in range(10):
            self.system.predict_person(test_embedding)
        
        # Benchmark prediction time
        start_time = time.time()
        for _ in range(num_iterations):
            self.system.predict_person(test_embedding)
        end_time = time.time()
        
        avg_prediction_time = (end_time - start_time) / num_iterations
        predictions_per_second = 1 / avg_prediction_time
        
        print(f"Average prediction time: {avg_prediction_time*1000:.2f}ms")
        print(f"Predictions per second: {predictions_per_second:.1f}")
        
        # Benchmark embedding extraction (if possible)
        # This would require a sample image - skip for now
        
        self.results['benchmark'] = {
            'avg_prediction_time': avg_prediction_time,
            'predictions_per_second': predictions_per_second
        }
        
        return avg_prediction_time
    
    def generate_performance_report(self, save_plots=True):
        """Generate comprehensive performance report"""
        print(f"\n{'='*50}")
        print(f"COMPREHENSIVE PERFORMANCE REPORT")
        print(f"{'='*50}")
        
        # System info
        print(f"System Configuration:")
        print(f"  Classifier: {self.system.classifier.get_name()}")
        print(f"  Parameters: {self.system.classifier.get_parameters()}")
        print(f"  Confidence Threshold: {self.system.confidence_threshold}")
        print(f"  Enrolled Users: {len(set(self.system.known_names))}")
        print(f"  Total Samples: {len(self.system.known_embeddings)}")
        
        # Run all analyses
        self.cross_validation_test()
        self.confidence_analysis()
        self.confusion_matrix_analysis()
        self.threshold_analysis()
        self.performance_benchmark()
        
        # Summary
        print(f"\n{'='*50}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        
        if 'overall_accuracy' in self.results:
            print(f"Overall Accuracy: {self.results['overall_accuracy']:.3f}")
        if 'avg_confidence' in self.results:
            print(f"Average Confidence: {self.results['avg_confidence']:.3f}")
        if 'cv_mean' in self.results:
            print(f"Cross-validation Accuracy: {self.results['cv_mean']:.3f} ± {self.results['cv_std']:.3f}")
        if 'avg_processing_time' in self.results:
            print(f"Processing Speed: {self.results['avg_processing_time']*1000:.1f}ms per face")
        
        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to JSON
        import json
        results_file = f"performance_analysis_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif key == 'classification_report':
                json_results[key] = value  # Already serializable
            else:
                json_results[key] = value
        
        try:
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {results_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        return self.results
    
    def plot_performance_charts(self):
        """Generate performance visualization charts"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Face Recognition Performance Analysis', fontsize=16)
            
            # 1. Confidence distribution
            if 'confidences' in self.results:
                ax1 = axes[0, 0]
                all_confidences = []
                labels = []
                
                for person, person_data in self.results['confidences'].items():
                    person_confidences = [item['confidence'] for item in person_data]
                    all_confidences.extend(person_confidences)
                    labels.extend([person] * len(person_confidences))
                
                df_conf = pd.DataFrame({'Confidence': all_confidences, 'Person': labels})
                sns.boxplot(data=df_conf, x='Person', y='Confidence', ax=ax1)
                ax1.set_title('Confidence Score Distribution by Person')
                ax1.tick_params(axis='x', rotation=45)
            
            # 2. Threshold analysis
            if 'threshold_analysis' in self.results:
                ax2 = axes[0, 1]
                thresholds = [item['threshold'] for item in self.results['threshold_analysis']]
                accuracies = [item['accuracy'] for item in self.results['threshold_analysis']]
                acceptance_rates = [item['acceptance_rate'] for item in self.results['threshold_analysis']]
                
                ax2.plot(thresholds, accuracies, 'b-o', label='Accuracy')
                ax2.plot(thresholds, acceptance_rates, 'r-s', label='Acceptance Rate')
                ax2.set_xlabel('Confidence Threshold')
                ax2.set_ylabel('Rate')
                ax2.set_title('Accuracy vs Acceptance Rate')
                ax2.legend()
                ax2.grid(True)
            
            # 3. Confusion Matrix
            if 'confusion_matrix' in self.results:
                ax3 = axes[1, 0]
                cm = self.results['confusion_matrix']
                labels = self.results['unique_labels']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=labels, yticklabels=labels, ax=ax3)
                ax3.set_title('Confusion Matrix')
                ax3.set_xlabel('Predicted')
                ax3.set_ylabel('Actual')
            
            # 4. Per-person accuracy
            if 'confidences' in self.results:
                ax4 = axes[1, 1]
                persons = []
                accuracies = []
                
                for person, person_data in self.results['confidences'].items():
                    accuracy = sum([item['correct'] for item in person_data]) / len(person_data)
                    persons.append(person)
                    accuracies.append(accuracy)
                
                bars = ax4.bar(persons, accuracies, color='skyblue')
                ax4.set_title('Per-Person Accuracy')
                ax4.set_ylabel('Accuracy')
                ax4.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{acc:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            plot_file = f"performance_charts_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Performance charts saved to: {plot_file}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"Error generating plots: {e}")

def analyze_system_performance(model_path='models/', db_path='attendance.db'):
    """Analyze performance of existing trained system"""
    
    # Load the trained system
    classifier = SVMClassifier()
    system = FaceRecognitionSystem(
        classifier=classifier,
        model_path=model_path,
        database_path=db_path
    )
    
    if not system.known_embeddings:
        print("No trained model found. Please enroll people first.")
        return None
    
    # Create analyzer and run analysis
    analyzer = PerformanceAnalyzer(system)
    results = analyzer.generate_performance_report()
    
    # Generate plots if possible
    analyzer.plot_performance_charts()
    
    return analyzer, results

if __name__ == "__main__":
    print("Starting Face Recognition Performance Analysis...")
    analyzer, results = analyze_system_performance()