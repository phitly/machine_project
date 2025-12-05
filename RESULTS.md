# Spam Email Detection - Results

## Dataset Information

### Training Data
- **Train Dataset 1**: 2,228 samples (2 columns)
  - Ham: 1,927 (86.49%)
  - Spam: 301 (13.51%)

- **Train Dataset 2**: 2,068 samples (2 columns)
  - Ham: 1,440 (69.63%)
  - Spam: 628 (30.37%)

- **Combined Training Data**: 4,296 samples
  - Ham: 3,367 (78.37%)
  - Spam: 929 (21.63%)

### Test Data
- **Test Dataset**: 6,447 samples (1 column)

## Data Preprocessing

1. **Text Cleaning**
   - Converted to lowercase
   - Removed URLs (http, www)
   - Removed email addresses
   - Removed phone numbers (10+ digits)
   - Removed special characters
   - Removed extra whitespace

2. **Feature Extraction**
   - **Method**: TF-IDF Vectorization
   - **Max Features**: 5,000
   - **N-gram Range**: (1, 2) - unigrams and bigrams
   - **Stop Words**: English
   - **Min Document Frequency**: 2
   - **Max Document Frequency**: 0.95
   - **Vocabulary Size**: 5,000 features

3. **Data Split**
   - Training set: 3,436 samples (80%)
   - Validation set: 860 samples (20%)
   - Stratified split to maintain class balance

## Model Training & Evaluation

### Models Tested

#### 1. Multinomial Naive Bayes
- **Validation Accuracy**: 94.53%
- **Precision**: 87.57%
- **Recall**: 87.10%
- **F1-Score**: 87.33%
- **Cross-Validation F1**: 88.07% (±0.85%)

**Classification Report:**
```
              precision    recall  f1-score   support
         Ham       0.96      0.97      0.97       674
        Spam       0.88      0.87      0.87       186
    accuracy                           0.95       860
```

#### 2. Logistic Regression
- **Validation Accuracy**: 92.67%
- **Precision**: 96.24%
- **Recall**: 68.82%
- **F1-Score**: 80.25%
- **Cross-Validation F1**: 81.57% (±1.92%)

**Classification Report:**
```
              precision    recall  f1-score   support
         Ham       0.92      0.99      0.96       674
        Spam       0.96      0.69      0.80       186
    accuracy                           0.93       860
```

#### 3. Linear SVC ⭐ **BEST MODEL**
- **Validation Accuracy**: 96.51%
- **Precision**: 96.43%
- **Recall**: 87.10%
- **F1-Score**: 91.53%
- **Cross-Validation F1**: 90.96% (±1.40%)

**Classification Report:**
```
              precision    recall  f1-score   support
         Ham       0.97      0.99      0.98       674
        Spam       0.96      0.87      0.92       186
    accuracy                           0.97       860
```

#### 4. Random Forest
- **Validation Accuracy**: 93.26%
- **Precision**: 96.38%
- **Recall**: 71.51%
- **F1-Score**: 82.10%
- **Cross-Validation F1**: 82.55% (±2.47%)

**Classification Report:**
```
              precision    recall  f1-score   support
         Ham       0.93      0.99      0.96       674
        Spam       0.96      0.72      0.82       186
    accuracy                           0.93       860
```

## Best Model Selection

**Selected Model**: Linear SVC

**Reasons**:
- Highest F1-Score (91.53%)
- Excellent balance between precision (96.43%) and recall (87.10%)
- Best cross-validation performance (90.96%)
- Lowest standard deviation in CV (±1.40%)
- Superior performance on both Ham and Spam classes

## Final Predictions

The best model (Linear SVC) was retrained on the full training dataset (4,296 samples) and used to predict on the test set.

### Test Set Predictions
- **Total Predictions**: 6,447
- **Predicted Ham (0)**: 5,208 (80.78%)
- **Predicted Spam (1)**: 1,239 (19.22%)

## Generated Files

1. ✅ `spam_predictions.csv` - Final predictions (0/1 format)
2. ✅ `class_distribution.png` - Training data class distribution visualization
3. ✅ `multinomial_naive_bayes_confusion_matrix.png` - Naive Bayes confusion matrix
4. ✅ `logistic_regression_confusion_matrix.png` - Logistic Regression confusion matrix
5. ✅ `linear_svc_confusion_matrix.png` - Linear SVC confusion matrix
6. ✅ `random_forest_confusion_matrix.png` - Random Forest confusion matrix

## Summary

- **Pipeline**: Data Loading → EDA → Preprocessing → Vectorization → Model Training → Evaluation → Prediction
- **Best Accuracy**: 96.51% on validation set
- **Model**: Linear Support Vector Classifier (Linear SVC)
- **Submission File**: `spam_predictions.csv` (6,447 predictions in 0/1 format)

**Status**: ✅ Ready for submission
