# Spam Email Detection - Project Report

## 1. Introduction

Email spam remains a significant challenge in modern communication systems. This project implements a machine learning-based approach to classify emails as either spam or ham (legitimate). Using supervised learning techniques on labeled training data, we developed a high-accuracy binary classifier capable of identifying spam emails with 96.51% accuracy.

## 2. Dataset Description

### 2.1 Training Data
The project utilized two separate training datasets:
- **Training Set 1**: 2,228 emails (86.49% ham, 13.51% spam)
- **Training Set 2**: 2,068 emails (69.63% ham, 30.37% spam)
- **Combined**: 4,296 total training samples with 3,367 ham and 929 spam emails

### 2.2 Test Data
- **Test Set**: 6,447 unlabeled email samples for prediction

### 2.3 Class Imbalance
The combined training set exhibits moderate class imbalance (21.63% spam vs 78.37% ham), which was addressed through stratified sampling during model validation.

## 3. Methodology

### 3.1 Data Preprocessing

**Text Cleaning Pipeline**:
1. **Lowercase Conversion**: Normalized all text to lowercase for consistency
2. **URL Removal**: Eliminated web links (http://, www.) to reduce noise
3. **Email Address Removal**: Stripped email addresses from message content
4. **Phone Number Removal**: Removed sequences of 10+ digits
5. **Special Character Removal**: Kept only alphabetic characters and spaces
6. **Whitespace Normalization**: Removed extra spaces and line breaks

This preprocessing ensures clean, standardized text input for feature extraction while removing spam-specific patterns that could cause overfitting.

### 3.2 Feature Engineering

**TF-IDF Vectorization**:
- **Method**: Term Frequency-Inverse Document Frequency
- **Vocabulary Size**: 5,000 features (optimized for performance vs. computational cost)
- **N-gram Range**: (1, 2) to capture both individual words and two-word phrases
- **Stop Words**: English stop words removed automatically
- **Min Document Frequency**: 2 (removes rare terms)
- **Max Document Frequency**: 0.95 (removes overly common terms)
- **Sublinear TF**: Applied to dampen the impact of term frequency

**Rationale**: TF-IDF effectively captures word importance while reducing the weight of common terms. Bigrams (2-word phrases) help identify spam patterns like "click here" or "free money."

### 3.3 Model Selection and Training

Four classification algorithms were trained and evaluated:

#### 1. Multinomial Naive Bayes
- **Assumption**: Features are conditionally independent
- **Strength**: Fast training, works well with text
- **Parameters**: Alpha=0.1 (Laplace smoothing)
- **Results**: 94.53% accuracy, 87.33% F1-score

#### 2. Logistic Regression
- **Assumption**: Linear decision boundary
- **Strength**: Interpretable, probabilistic output
- **Parameters**: C=1.0, max_iter=1000
- **Results**: 92.67% accuracy, 80.25% F1-score

#### 3. Linear Support Vector Classifier (SVC) ⭐
- **Assumption**: Linear separability in high-dimensional space
- **Strength**: Effective for high-dimensional text data
- **Parameters**: C=1.0, max_iter=1000
- **Results**: **96.51% accuracy, 91.53% F1-score**

#### 4. Random Forest
- **Assumption**: Ensemble of decision trees
- **Strength**: Handles non-linearity, feature importance
- **Parameters**: n_estimators=100, max_depth=50
- **Results**: 93.26% accuracy, 82.10% F1-score

### 3.4 Model Evaluation

**Validation Strategy**:
- 80/20 train-validation split with stratification
- 5-fold cross-validation for robustness assessment
- Metrics: Accuracy, Precision, Recall, F1-Score

**Evaluation Metrics**:
- **Accuracy**: Overall correct predictions
- **Precision**: Proportion of predicted spam that is actually spam
- **Recall**: Proportion of actual spam correctly identified
- **F1-Score**: Harmonic mean of precision and recall (balances both)

**Linear SVC Performance** (Best Model):
```
              Precision    Recall    F1-Score    Support
Ham              97%        99%        98%        674
Spam             96%        87%        92%        186
Accuracy                               97%        860
```

**Cross-Validation**: Mean F1 = 90.96% (±1.40%) demonstrates consistent performance across different data splits.

## 4. Results

### 4.1 Model Comparison

| Model                    | Accuracy | Precision | Recall | F1-Score | CV F1 (±std) |
|--------------------------|----------|-----------|--------|----------|--------------|
| Multinomial Naive Bayes  | 94.53%   | 87.57%    | 87.10% | 87.33%   | 88.07% (±0.85%) |
| Logistic Regression      | 92.67%   | 96.24%    | 68.82% | 80.25%   | 81.57% (±1.92%) |
| **Linear SVC** ⭐        | **96.51%** | **96.43%** | **87.10%** | **91.53%** | **90.96% (±1.40%)** |
| Random Forest            | 93.26%   | 96.38%    | 71.51% | 82.10%   | 82.55% (±2.47%) |

### 4.2 Best Model Selection

**Linear SVC** was selected as the final model based on:
1. Highest F1-score (91.53%) - best balance of precision and recall
2. Superior validation accuracy (96.51%)
3. Excellent cross-validation performance with low variance (±1.40%)
4. High precision (96.43%) minimizes false positives
5. Strong recall (87.10%) catches most spam emails

### 4.3 Test Set Predictions

After retraining on the full 4,296 training samples, the Linear SVC model predicted:
- **Ham (0)**: 5,208 emails (80.78%)
- **Spam (1)**: 1,239 emails (19.22%)

This distribution aligns reasonably with the training set spam rate (21.63%), suggesting the model generalizes well.

## 5. Implementation Details

### 5.1 Technology Stack
- **Language**: Python 3.13
- **Libraries**: 
  - scikit-learn (machine learning)
  - pandas (data manipulation)
  - numpy (numerical computing)
  - matplotlib & seaborn (visualization)

### 5.2 Code Structure
```
spam_email_classifier.py
├── SpamEmailDetector class
│   ├── load_data()           # Load and standardize datasets
│   ├── preprocess_text()     # Clean email text
│   ├── explore_data()        # EDA and visualization
│   ├── prepare_data()        # Combine and process data
│   ├── vectorize_text()      # TF-IDF feature extraction
│   ├── train_and_evaluate_models()  # Train 4 models
│   ├── predict_test()        # Generate predictions
│   └── save_predictions()    # Export to CSV
└── main()                    # Pipeline execution
```

### 5.3 Reproducibility
- Random seed: 42 (ensures consistent results)
- Stratified splits maintain class distribution
- All preprocessing steps are deterministic

## 6. Challenges and Solutions

### 6.1 Class Imbalance
**Challenge**: 78% ham vs 22% spam could bias the model toward predicting ham.

**Solution**: 
- Stratified train-validation split
- F1-score as primary metric (balances precision/recall)
- Cross-validation to ensure robust evaluation

### 6.2 Overfitting Risk
**Challenge**: High-dimensional features (5,000) with limited samples (4,296).

**Solution**:
- TF-IDF max_df=0.95 removes overly common features
- Min_df=2 removes rare terms
- Regularization in Linear SVC (C=1.0)
- Cross-validation to detect overfitting

### 6.3 Different Dataset Formats
**Challenge**: Training sets had different column names and structures.

**Solution**: Automated column detection and standardization in `load_data()` method.

## 7. Conclusions

This project successfully developed a spam email classifier achieving **96.51% accuracy** using Linear Support Vector Classification. Key findings:

1. **TF-IDF with bigrams** effectively captures spam characteristics
2. **Linear SVC outperforms** other algorithms for this text classification task
3. **Balanced metrics** (96% precision, 87% recall) ensure both low false positives and good spam detection
4. **Consistent cross-validation** (±1.40% std) indicates robust generalization

The final model successfully classified 6,447 test emails with predictions distributed similarly to the training data, suggesting good generalization to unseen data.

### Future Improvements
- Deep learning models (LSTM, BERT) for semantic understanding
- Ensemble methods combining multiple classifiers
- Additional features (email metadata, sender information)
- Real-time retraining pipeline for evolving spam patterns
- Hyperparameter tuning via grid search

---

**Project Files**:
- `spam_email_classifier.py` - Main implementation
- `spam_predictions.csv` - Test set predictions (6,447 samples)
- `RESULTS.md` - Detailed results and metrics
- Confusion matrix visualizations (4 models)
- Class distribution plots

**Performance Summary**: 96.51% accuracy, 91.53% F1-score, ready for deployment.
