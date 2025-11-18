

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class SpamDetector:
    """Email Spam Detection Classifier"""
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        
    def load_data(self, train1_path, train1_labels_path, 
                  train2_path, train2_labels_path, test_path):
        """Load all datasets"""
        print("Loading datasets...")
        
        # Load training data 1
        self.train1_data = pd.read_csv(train1_path)
        self.train1_labels = pd.read_csv(train1_labels_path)
        
        # Load training data 2
        self.train2_data = pd.read_csv(train2_path)
        self.train2_labels = pd.read_csv(train2_labels_path)
        
        # Load test data
        self.test_data = pd.read_csv(test_path)
        
        print(f"Train Data 1: {self.train1_data.shape}")
        print(f"Train Data 2: {self.train2_data.shape}")
        print(f"Test Data: {self.test_data.shape}")
        
        return self
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Combine datasets for analysis
        train1_combined = self.train1_data.copy()
        train1_combined['label'] = self.train1_labels.iloc[:, 0]
        
        train2_combined = self.train2_data.copy()
        train2_combined['label'] = self.train2_labels.iloc[:, 0]
        
        print("\nTrainData 1 Label Distribution:")
        print(train1_combined['label'].value_counts())
        
        print("\nTrainData 2 Label Distribution:")
        print(train2_combined['label'].value_counts())
        
        # Visualize class distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        train1_combined['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
        axes[0].set_title('TrainData 1 - Class Distribution')
        axes[0].set_xlabel('Label')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels(['Ham', 'Spam'], rotation=0)
        
        train2_combined['label'].value_counts().plot(kind='bar', ax=axes[1], color=['green', 'red'])
        axes[1].set_title('TrainData 2 - Class Distribution')
        axes[1].set_xlabel('Label')
        axes[1].set_ylabel('Count')
        axes[1].set_xticklabels(['Ham', 'Spam'], rotation=0)
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        print("\nClass distribution plot saved as 'class_distribution.png'")
        
        return self
    
    def prepare_data(self):
        """Combine training datasets and prepare for modeling"""
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        
        # Combine training data
        train1_combined = self.train1_data.copy()
        train1_combined['label'] = self.train1_labels.iloc[:, 0]
        
        train2_combined = self.train2_data.copy()
        train2_combined['label'] = self.train2_labels.iloc[:, 0]
        
        # Merge both training sets
        self.train_combined = pd.concat([train1_combined, train2_combined], ignore_index=True)
        
        print(f"\nCombined Training Data: {self.train_combined.shape}")
        print(f"\nFinal Label Distribution:")
        print(self.train_combined['label'].value_counts())
        
        # Get text column name (assuming it's the first column or contains 'text', 'email', 'message')
        text_columns = [col for col in self.train_combined.columns if col != 'label']
        if len(text_columns) > 0:
            self.text_column = text_columns[0]
        else:
            self.text_column = self.train_combined.columns[0]
        
        print(f"\nUsing column '{self.text_column}' as email text")
        
        # Separate features and labels
        self.X_train = self.train_combined[self.text_column]
        self.y_train = self.train_combined['label']
        
        # Get test column
        test_text_columns = [col for col in self.test_data.columns]
        self.X_test = self.test_data[test_text_columns[0]]
        
        return self
    
    def preprocess_and_vectorize(self, max_features=5000):
        """Vectorize text data using TF-IDF"""
        print("\n" + "="*50)
        print("TEXT VECTORIZATION")
        print("="*50)
        
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit and transform training data
        print(f"\nVectorizing with max_features={max_features}...")
        self.X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vectorized = self.vectorizer.transform(self.X_test)
        
        print(f"Training data shape after vectorization: {self.X_train_vectorized.shape}")
        print(f"Test data shape after vectorization: {self.X_test_vectorized.shape}")
        
        return self
    
    def train_models(self):
        """Train multiple classifiers and compare performance"""
        print("\n" + "="*50)
        print("MODEL TRAINING & EVALUATION")
        print("="*50)
        
        # Split data for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train_vectorized, self.y_train, 
            test_size=0.2, random_state=42, stratify=self.y_train
        )
        
        # Define models to test
        models = {
            'Multinomial Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            
            # Train model
            model.fit(X_tr, y_tr)
            
            # Validate
            y_pred = model.predict(X_val)
            
            # Metrics
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_vectorized, self.y_train, cv=5, scoring='f1_weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Validation F1-Score: {f1:.4f}")
            print(f"Cross-Validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print("\nClassification Report:")
            print(classification_report(y_val, y_pred))
            
            # Confusion Matrix
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            plt.title(f'{name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'{name.replace(" ", "_").lower()}_confusion_matrix.png', dpi=300)
            print(f"Confusion matrix saved as '{name.replace(' ', '_').lower()}_confusion_matrix.png'")
        
        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        self.model = results[best_model_name]['model']
        
        print("\n" + "="*50)
        print(f"BEST MODEL: {best_model_name}")
        print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
        print("="*50)
        
        # Retrain best model on full training data
        print(f"\nRetraining {best_model_name} on full training data...")
        self.model.fit(self.X_train_vectorized, self.y_train)
        
        return self
    
    def predict_test(self):
        """Make predictions on test data"""
        print("\n" + "="*50)
        print("PREDICTION ON TEST DATA")
        print("="*50)
        
        # Predict
        self.test_predictions = self.model.predict(self.X_test_vectorized)
        
        print(f"\nTotal predictions: {len(self.test_predictions)}")
        print(f"Predicted Spam: {sum(self.test_predictions == 1)}")
        print(f"Predicted Ham: {sum(self.test_predictions == 0)}")
        
        return self
    
    def save_predictions(self, output_file='predictions.csv'):
        """Save predictions to CSV file"""
        print("\n" + "="*50)
        print("SAVING PREDICTIONS")
        print("="*50)
        
        
        submission = pd.DataFrame({
            'Id': range(len(self.test_predictions)),
            'Label': self.test_predictions
        })
        
        
        submission.to_csv(output_file, index=False)
        print(f"\nPredictions saved to '{output_file}'")
        print(f"Total samples: {len(submission)}")
        
        return self


def main():
    """Main execution function"""
    print("="*50)
    print("SPAM EMAIL DETECTION CLASSIFIER")
    print("="*50)
    
    
    detector = SpamDetector()
    
    
    detector.load_data(
        train1_path='train_data1.csv',
        train1_labels_path='train_labels1.csv',
        train2_path='train_data2.csv',
        train2_labels_path='train_labels2.csv',
        test_path='test_data.csv'
    )
    
    # Execute pipeline
    (detector
     .explore_data()
     .prepare_data()
     .preprocess_and_vectorize(max_features=5000)
     .train_models()
     .predict_test()
     .save_predictions('spam_predictions.csv'))
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)


if __name__ == "__main__":
    main()
