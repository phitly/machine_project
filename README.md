# Spam Email Detection Classifier

A machine learning-based email spam detection system that classifies emails as **Spam** or **Ham** (legitimate emails).

## 📊 Dataset Information

- **Training Dataset 1**: 2,228 samples (1,927 Ham, 301 Spam - 86.5% Ham, 13.5% Spam)
- **Training Dataset 2**: 2,068 samples (1,440 Ham, 628 Spam - 69.6% Ham, 30.4% Spam)
- **Combined Training**: 4,296 samples (3,367 Ham, 929 Spam - 78.4% Ham, 21.6% Spam)
- **Test Dataset**: 6,447 samples

## 🎯 Model Performance

### Best Model: **Linear SVC**

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.51% |
| **Precision** | 96.43% |
| **Recall** | 87.10% |
| **F1-Score** | 91.53% |
| **Cross-Validation F1** | 90.96% (±1.40%) |

### All Models Tested

1. **Linear SVC** ⭐ (Best)
   - Accuracy: 96.51%, F1: 91.53%
   
2. **Multinomial Naive Bayes**
   - Accuracy: 94.53%, F1: 87.33%
   
3. **Random Forest**
   - Accuracy: 93.26%, F1: 82.10%
   
4. **Logistic Regression**
   - Accuracy: 92.67%, F1: 80.25%

## 🚀 Quick Start

### 1. Install Dependencies

The project uses a virtual environment. Dependencies are already installed:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn





## 📁 Project Structure

```
ML_project/
├── spam_email_classifier.py          # Main classifier script
├── Spam Email Detection Train 1.csv  # Training dataset 1
├── Spam Email Detection Train 2.csv  # Training dataset 2
├── Spam Email Detection Test.csv     # Test dataset
├── spam_predictions.csv              # Output predictions ✓
├── class_distribution.png            # Class distribution plots ✓
├── multinomial_naive_bayes_confusion_matrix.png
├── logistic_regression_confusion_matrix.png
├── linear_svc_confusion_matrix.png
├── random_forest_confusion_matrix.png
├── requirements.txt
└── README.md
```

## 🔍 Features

### Text Preprocessing
- Lowercase conversion
- URL removal
- Email address removal
- Phone number removal
- Special character cleaning
- Whitespace normalization

### Feature Extraction
- **TF-IDF Vectorization**
  - Max features: 5,000
  - N-gram range: (1, 2) - unigrams and bigrams
  - Stop words: English
  - Sublinear TF scaling
  - Min document frequency: 2
  - Max document frequency: 95%

### Model Training
- Train/validation split: 80/20
- Stratified sampling to maintain class distribution
- 5-fold cross-validation
- Multiple algorithms tested and compared

## 📈 Results

### Test Set Predictions
- **Ham emails**: 5,208 (80.78%)
- **Spam emails**: 1,239 (19.22%)

### Sample Predictions

```
1. [SPAM] SMS SERVICES. for your inclusive text credits...
2. [HAM] ILL B DOWN SOON
3. [HAM] Subject: vastar / big thicket...
4. [SPAM] Subject: skip the doctor - buy meds online...
5. [HAM] Subject: credit watch list - - week of 10 / 29 / 01...
```

## 🛠️ How It Works

1. **Data Loading**: Loads and standardizes both training datasets
2. **EDA**: Analyzes class distribution and generates visualizations
3. **Preprocessing**: Cleans email text and removes noise
4. **Vectorization**: Converts text to TF-IDF feature vectors
5. **Training**: Trains 4 different ML models with cross-validation
6. **Selection**: Automatically selects best performing model (Linear SVC)
7. **Prediction**: Generates predictions on 6,447 test emails
8. **Export**: Saves results to `spam_predictions.csv`

## 📊 Visualizations

All visualizations are automatically generated and saved:

1. **class_distribution.png** - Shows ham/spam distribution for both training sets
2. **Confusion matrices** - One for each model showing true/false positives/negatives

## 🎓 Key Insights

- **Linear SVC** performed best with 96.51% accuracy
- **Class imbalance** handled through stratified sampling
- **TF-IDF** features with bigrams capture important spam patterns
- **High precision** (96.43%) means few false positives - legitimate emails rarely marked as spam
- **Good recall** (87.10%) means most spam is caught
