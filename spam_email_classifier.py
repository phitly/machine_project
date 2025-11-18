import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             f1_score, precision_score, recall_score)
import re
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class SpamEmailDetector:
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.label_mapping = {'ham': 0, 'spam': 1}
        
    def load_data(self, train1_path, train2_path, test_path):
        print("="*70)
        print("LOADING DATASETS")
        print("="*70)
        
        self.train1 = pd.read_csv(train1_path)
        self.train2 = pd.read_csv(train2_path)
        self.test = pd.read_csv(test_path)
        
        if 'v1' in self.train1.columns and 'v2' in self.train1.columns:
            self.train1 = self.train1[['v1', 'v2']].copy()
            self.train1.columns = ['label', 'message']
        
        if 'label' in self.train2.columns and 'text' in self.train2.columns:
            self.train2 = self.train2[['label', 'text']].copy()
            self.train2.columns = ['label', 'message']
        
        if 'message' in self.test.columns:
            self.test = self.test[['message']].copy()
        else:
            col_name = self.test.columns[0]
            self.test = self.test[[col_name]].copy()
            self.test.columns = ['message']
        
        print(f"\nTrain Dataset 1 shape: {self.train1.shape}")
        print(f"Train Dataset 2 shape: {self.train2.shape}")
        print(f"Test Dataset shape: {self.test.shape}")
        
        print("\nTrain Dataset 1 - Sample:")
        print(self.train1.head(2))
        print("\nTrain Dataset 2 - Sample:")
        print(self.train2.head(2))
        
        return self
    
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\d{10,}', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text
    
    def explore_data(self):
        print("\n" + "="*70)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        print(f"\nTrain Dataset 1 - Label Distribution:")
        print(self.train1['label'].value_counts())
        spam_count1 = (self.train1['label'] == 'spam').sum()
        ham_count1 = (self.train1['label'] == 'ham').sum()
        print(f"  Ham: {ham_count1} ({ham_count1/len(self.train1)*100:.2f}%)")
        print(f"  Spam: {spam_count1} ({spam_count1/len(self.train1)*100:.2f}%)")
        
        print(f"\nTrain Dataset 2 - Label Distribution:")
        print(self.train2['label'].value_counts())
        spam_count2 = (self.train2['label'] == 'spam').sum()
        ham_count2 = (self.train2['label'] == 'ham').sum()
        print(f"  Ham: {ham_count2} ({ham_count2/len(self.train2)*100:.2f}%)")
        print(f"  Spam: {spam_count2} ({spam_count2/len(self.train2)*100:.2f}%)")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        train1_counts = self.train1['label'].value_counts()
        axes[0].bar(train1_counts.index, train1_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Train Dataset 1 - Class Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(train1_counts.values):
            axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        train2_counts = self.train2['label'].value_counts()
        axes[1].bar(train2_counts.index, train2_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[1].set_title('Train Dataset 2 - Class Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Class', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(train2_counts.values):
            axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: 'class_distribution.png'")
        plt.close()
        
        return self
    
    def prepare_data(self):
        print("\n" + "="*70)
        print("DATA PREPARATION")
        print("="*70)
        
        self.train_combined = pd.concat([self.train1, self.train2], ignore_index=True)
        
        print(f"\nCombined Training Data: {self.train_combined.shape}")
        
        print(f"\nCombined Label Distribution:")
        print(self.train_combined['label'].value_counts())
        
        print("\nPreprocessing text data...")
        self.train_combined['cleaned_text'] = self.train_combined['message'].apply(self.preprocess_text)
        self.test['cleaned_text'] = self.test['message'].apply(self.preprocess_text)
        
        self.train_combined['label_binary'] = self.train_combined['label'].map(self.label_mapping)
        
        self.X_train = self.train_combined['cleaned_text']
        self.y_train = self.train_combined['label_binary']
        self.X_test = self.test['cleaned_text']
        
        print(f"✓ Text preprocessing complete")
        print(f"✓ Training samples: {len(self.X_train)}")
        print(f"✓ Test samples: {len(self.X_test)}")
        
        return self
    
    def vectorize_text(self, max_features=5000):
        print("\n" + "="*70)
        print("TEXT VECTORIZATION (TF-IDF)")
        print("="*70)
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            sublinear_tf=True
        )
        
        print(f"\nVectorizing text with max_features={max_features}...")
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        
        print(f"✓ Training data shape: {self.X_train_vec.shape}")
        print(f"✓ Test data shape: {self.X_test_vec.shape}")
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return self
    
    def train_and_evaluate_models(self):
        print("\n" + "="*70)
        print("MODEL TRAINING & EVALUATION")
        print("="*70)
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train_vec, self.y_train,
            test_size=0.2,
            random_state=42,
            stratify=self.y_train
        )
        
        print(f"\nTraining set: {X_tr.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        
        models = {
            'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            'Linear SVC': LinearSVC(C=1.0, max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*70}")
            print(f"Training: {name}")
            print(f"{'='*70}")
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, self.X_train_vec, self.y_train, 
                                       cv=cv, scoring='f1')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"\nValidation Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"\nCross-Validation (5-fold):")
            print(f"  Mean F1:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            print(f"\nClassification Report:")
            print(classification_report(y_val, y_pred, target_names=['Ham', 'Spam']))
            
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Ham', 'Spam'],
                       yticklabels=['Ham', 'Spam'],
                       cbar_kws={'label': 'Count'})
            plt.title(f'{name}\nConfusion Matrix', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            filename = f"{name.replace(' ', '_').lower()}_confusion_matrix.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
            plt.close()
        
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        self.model = results[best_model_name]['model']
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_model_name}")
        print("="*70)
        print(f"Accuracy:  {results[best_model_name]['accuracy']:.4f}")
        print(f"Precision: {results[best_model_name]['precision']:.4f}")
        print(f"Recall:    {results[best_model_name]['recall']:.4f}")
        print(f"F1-Score:  {results[best_model_name]['f1_score']:.4f}")
        print(f"CV F1:     {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std']:.4f})")
        
        print(f"\n✓ Retraining {best_model_name} on full training data...")
        self.model.fit(self.X_train_vec, self.y_train)
        print(f"✓ Training complete!")
        
        self.results = results
        self.best_model_name = best_model_name
        
        return self
    
    def predict_test(self):
        print("\n" + "="*70)
        print("PREDICTION ON TEST DATA")
        print("="*70)
        
        self.test_predictions = self.model.predict(self.X_test_vec)
        
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        self.test_predictions_labels = [reverse_mapping[pred] for pred in self.test_predictions]
        
        spam_count = sum(self.test_predictions == 1)
        ham_count = sum(self.test_predictions == 0)
        
        print(f"\nTotal predictions: {len(self.test_predictions)}")
        print(f"Predicted Ham:  {ham_count} ({ham_count/len(self.test_predictions)*100:.2f}%)")
        print(f"Predicted Spam: {spam_count} ({spam_count/len(self.test_predictions)*100:.2f}%)")
        
        return self
    
    def save_predictions(self, output_file='spam_predictions.csv'):
        print("\n" + "="*70)
        print("SAVING PREDICTIONS")
        print("="*70)
        
        submission = pd.DataFrame({
            'message': self.test['message'],
            'prediction': self.test_predictions_labels
        })
        
        submission.to_csv(output_file, index=False)
        print(f"\n✓ Predictions saved to: '{output_file}'")
        print(f"✓ Total samples: {len(submission)}")
        
        print(f"\nSample Predictions (first 10):")
        print("="*70)
        for idx in range(min(10, len(submission))):
            msg = submission.iloc[idx]['message']
            pred = submission.iloc[idx]['prediction']
            msg_short = msg[:60] + "..." if len(str(msg)) > 60 else msg
            print(f"{idx+1}. [{pred.upper()}] {msg_short}")
        
        return self
    
    def generate_summary(self):
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"\nModel Performance:")
        for metric, value in self.results[self.best_model_name].items():
            if metric != 'model':
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        print(f"\nDataset Statistics:")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        print(f"\nTest Predictions:")
        spam_count = sum(self.test_predictions == 1)
        ham_count = sum(self.test_predictions == 0)
        print(f"  Ham: {ham_count} ({ham_count/len(self.test_predictions)*100:.2f}%)")
        print(f"  Spam: {spam_count} ({spam_count/len(self.test_predictions)*100:.2f}%)")
        
        print(f"\nGenerated Files:")
        print(f"  ✓ class_distribution.png")
        print(f"  ✓ confusion matrices for all models")
        print(f"  ✓ spam_predictions.csv")


def main():
    print("\n" + "="*70)
    print(" "*15 + "SPAM EMAIL DETECTION CLASSIFIER")
    print("="*70)
    
    detector = SpamEmailDetector()
    
    (detector
     .load_data(
         train1_path='Spam Email Detection Train 1.csv',
         train2_path='Spam Email Detection Train 2.csv',
         test_path='Spam Email Detection Test.csv'
     )
     .explore_data()
     .prepare_data()
     .vectorize_text(max_features=5000)
     .train_and_evaluate_models()
     .predict_test()
     .save_predictions('spam_predictions.csv')
     .generate_summary())
    
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
