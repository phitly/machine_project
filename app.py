import gradio as gr
import pandas as pd
from spam_email_classifier import SpamEmailDetector
import os


class GradioSpamDetectorInterface:
    """Gradio interface for Spam Email Detector"""
    
    def __init__(self):
        self.detector = None
        self.model_trained = False
        
    def train_model(self, train1_file, train2_file, test_file):
        """Train the model using uploaded files"""
        try:
            train1_path = train1_file if isinstance(train1_file, str) else train1_file.name
            train2_path = train2_file if isinstance(train2_file, str) else train2_file.name
            test_path = test_file if isinstance(test_file, str) else test_file.name
            
            self.detector = SpamEmailDetector()
            
            (self.detector
             .load_data(train1_path, train2_path, test_path)
             .explore_data()
             .prepare_data()
             .vectorize_text(max_features=5000)
             .train_and_evaluate_models()
             .predict_test()
             .save_predictions('spam_predictions.csv'))
            
            self.model_trained = True
            
            results_text = f"✓ Model trained successfully!\n\n"
            results_text += f"Best Model: {self.detector.best_model_name}\n\n"
            results_text += "Model Performance:\n"
            results_text += "-" * 50 + "\n"
            
            for metric, value in self.detector.results[self.detector.best_model_name].items():
                if metric != 'model':
                    if isinstance(value, float):
                        results_text += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
            
            results_text += "-" * 50 + "\n"
            results_text += f"Predictions saved to 'spam_predictions.csv'"
            
            return results_text
            
        except Exception as e:
            return f"Error training model: {str(e)}"
    
    def predict_single_email(self, email_text):
        """Predict spam/ham for a single email"""
        if not self.model_trained or self.detector is None:
            return "Model not trained yet. Please train the model first."
        
        try:
            cleaned_text = self.detector.preprocess_text(email_text)
            
            email_vectorized = self.detector.vectorizer.transform([cleaned_text])
            
            prediction = self.detector.model.predict(email_vectorized)[0]
            
            if hasattr(self.detector.model, 'predict_proba'):
                probabilities = self.detector.model.predict_proba(email_vectorized)[0]
                ham_prob = probabilities[0] * 100
                spam_prob = probabilities[1] * 100
            else:
                ham_prob = (1 - prediction) * 100 if prediction < 1 else 0
                spam_prob = prediction * 100 if prediction > 0 else 100
            
            label = "SPAM" if prediction == 1 else "✓ HAM"
            confidence = max(ham_prob, spam_prob)
            
            result = f"""
**Classification: {label}**

Confidence: {confidence:.2f}%

Probabilities:
- Ham: {ham_prob:.2f}%
- Spam: {spam_prob:.2f}%

Model: {self.detector.best_model_name}
            """
            return result
            
        except Exception as e:
            return f"Error making prediction: {str(e)}"
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.model_trained or self.detector is None:
            return "Model not trained yet."
        
        try:
            info = f"""
## Trained Model Information

**Model Details:**
- Model Type: {self.detector.best_model_name}
- Vectorizer: TF-IDF
- Max Features: 5000
- N-gram Range: (1, 2)
- Stop Words: English

**Training Data:**
- Combined Training Samples: {len(self.detector.train_combined)}
- Feature Dimension: {self.detector.X_train_vec.shape[1]}
- Classes: Ham (0), Spam (1)

**Performance Metrics (Best Model):**
"""
            for metric, value in self.detector.results[self.detector.best_model_name].items():
                if metric != 'model' and isinstance(value, float):
                    info += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
            
            return info
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def load_example_predictions(self):
        """Load and display example predictions from saved file"""
        try:
            if os.path.exists('spam_predictions.csv'):
                predictions_df = pd.read_csv('spam_predictions.csv', header=None)
                summary = f"""
**Predictions Summary:**
- Total Predictions: {len(predictions_df)}
- Spam Count: {(predictions_df[0] == 1).sum()}
- Ham Count: {(predictions_df[0] == 0).sum()}

Prediction distribution:
- Spam: {((predictions_df[0] == 1).sum() / len(predictions_df) * 100):.2f}%
- Ham: {((predictions_df[0] == 0).sum() / len(predictions_df) * 100):.2f}%
                """
                return summary
            else:
                return " No predictions file found. Train the model first."
        except Exception as e:
            return f"Error loading predictions: {str(e)}"


interface = GradioSpamDetectorInterface()

with gr.Blocks(title="Spam Email Detection System") as demo:
    gr.Markdown("# Spam Email Detection System")
    gr.Markdown("""
    A machine learning application for detecting spam emails.
    """)
    
    with gr.Tabs():
        with gr.TabItem("🎯 Training", id="training"):
            gr.Markdown("### Upload Training Data")
            gr.Markdown("Upload your training CSV files to train the spam detection model.")
            
            with gr.Row():
                with gr.Column():
                    train1_input = gr.File(
                        label="Training Data 1",
                        file_count="single",
                        file_types=[".csv"]
                    )
                
                with gr.Column():
                    train2_input = gr.File(
                        label="Training Data 2",
                        file_count="single",
                        file_types=[".csv"]
                    )
            
            test_input = gr.File(
                label="Test Data",
                file_count="single",
                file_types=[".csv"]
            )
            
            with gr.Row():
                train_button = gr.Button("🚀 Train Model", variant="primary", size="lg")
                clear_button = gr.Button("🔄 Clear", variant="secondary", size="lg")
            
            train_output = gr.Textbox(
                label="Training Status",
                interactive=False,
                lines=10,
                max_lines=15
            )
            
            train_button.click(
                fn=interface.train_model,
                inputs=[train1_input, train2_input, test_input],
                outputs=train_output
            )
            
            clear_button.click(
                fn=lambda: ("", "", "", ""),
                outputs=[train1_input, train2_input, test_input, train_output]
            )
        
        with gr.TabItem("🔍 Test Email", id="prediction"):
            gr.Markdown("### Classify Individual Emails")
            gr.Markdown("Enter or paste an email to classify it as spam or ham.")
            
            email_input = gr.Textbox(
                label="Email Content",
                lines=8,
                max_lines=15,
                placeholder="Paste the email text here..."
            )
            
            with gr.Row():
                predict_button = gr.Button("📬 Classify Email", variant="primary", size="lg")
                clear_email_button = gr.Button("🔄 Clear", variant="secondary", size="lg")
            
            predict_output = gr.Markdown(label="Prediction Result")
            
            predict_button.click(
                fn=interface.predict_single_email,
                inputs=email_input,
                outputs=predict_output
            )
            
            clear_email_button.click(
                fn=lambda: ("", ""),
                outputs=[email_input, predict_output]
            )
        
        with gr.TabItem("Model Info", id="info"):
            gr.Markdown("### Trained Model Details")
            
            with gr.Row():
                info_button = gr.Button("📊 Show Model Info", variant="primary", size="lg")
                predictions_button = gr.Button("📈 Show Predictions Summary", variant="secondary", size="lg")
            
            info_output = gr.Markdown(label="Model Information")
            
            info_button.click(
                fn=interface.get_model_info,
                outputs=info_output
            )
            
            predictions_button.click(
                fn=interface.load_example_predictions,
                outputs=info_output
            )
        
        with gr.TabItem("📚 Instructions", id="instructions"):
            gr.Markdown("""
## How to Use This Application

### Step 1: Training
1. Go to the **Training** tab
2. Upload your training CSV files:
   - **Training Data 1**: First dataset (like, `Spam Email Detection Train 1.csv`)
   - **Training Data 2**: Second dataset (like, `Spam Email Detection Train 2.csv`)
   - **Test Data**: Test dataset for predictions (like, `Spam Email Detection Test.csv`)
3. Click **Train Model** button
4. Wait for the model to train and evaluate

### Step 2: Test Individual Emails
1. Go to the **Test Email** tab
2. Paste or type an email content
3. Click **Classify Email** button
4. View the prediction (SPAM or HAM) with confidence score

### Step 3: View Model Information
1. Go to the **Model Info** tab
2. Click **Show Model Info** to see model details and performance metrics
3. Click **Show Predictions Summary** to see test predictions statistics

## Supported File Formats
- CSV files with the following columns:
  - **Train 1**: `v1` (label), `v2` (message)
  - **Train 2**: `label`, `text`
  - **Test**: `message` (or any message column)

## Features
- ✓ Multiple classification algorithms
- ✓ TF-IDF text vectorization
- ✓ Cross-validation and performance metrics
- ✓ Confusion matrices and visualizations
- ✓ Single email classification
- ✓ Batch predictions

## Models Used
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVC
- Random Forest

The best performing model is automatically selected for predictions.
            """)


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
