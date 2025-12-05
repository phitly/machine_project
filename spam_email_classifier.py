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
        plt.close()
        
        return self
    
    def prepare_data(self):
        self.train_combined = pd.concat([self.train1, self.train2], ignore_index=True)
        self.train_combined['cleaned_text'] = self.train_combined['message'].apply(self.preprocess_text)
        self.test['cleaned_text'] = self.test['message'].apply(self.preprocess_text)
        self.train_combined['label_binary'] = self.train_combined['label'].map(self.label_mapping)
        
        self.X_train = self.train_combined['cleaned_text']
        self.y_train = self.train_combined['label_binary']
        self.X_test = self.test['cleaned_text']
        
        return self
    
    def vectorize_text(self, max_features=5000):
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
        
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        
        return self
    
    def train_and_evaluate_models(self):
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train_vec, self.y_train,
            test_size=0.2,
            random_state=42,
            stratify=self.y_train
        )
        
        models = {
            'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            'Linear SVC': LinearSVC(C=1.0, max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for name, model in models.items():
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
            plt.close()
        
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        self.model = results[best_model_name]['model']
        self.model.fit(self.X_train_vec, self.y_train)
        
        self.results = results
        self.best_model_name = best_model_name
        
        return self
    
    def predict_test(self):
        self.test_predictions = self.model.predict(self.X_test_vec)
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        self.test_predictions_labels = [reverse_mapping[pred] for pred in self.test_predictions]
        return self
    
    def save_predictions(self, output_file='spam_predictions.csv'):
        pd.DataFrame({'prediction': self.test_predictions}).to_csv(output_file, index=False, header=False)
        return self
    
    def generate_summary(self):
        pass


def main():
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


if __name__ == "__main__":
    main()
