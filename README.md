# DistilBERT Sentiment Analysis on IMDb Dataset
  
This repository contains the implementation of a sentiment analysis pipeline using the DistilBERT transformer model. The task is performed on a subset of the IMDb movie review dataset, focusing on efficient training and evaluation of the model.

## **Project Overview**
- **Objective**: Perform binary sentiment classification (Positive vs. Negative) on IMDb movie reviews using the DistilBERT transformer model.
- **Dataset**: IMDb dataset from the `datasets` library.
- **Evaluation Metrics**:
  - F1 Score: 0.5061
  - Accuracy: 59.40%
  - AUC-ROC: 0.6032

## **Features**
1. **Data Preparation**:
   - Loaded IMDb dataset using the `datasets` library.
   - Preprocessed and tokenized the dataset using `DistilBertTokenizerFast`.
   - Selected smaller subsets of the dataset for faster training and evaluation.
   
2. **Model Training**:
   - Used `DistilBertForSequenceClassification` with 2 output labels (Positive, Negative).
   - Configured `TrainingArguments` for CPU training with reduced steps and epochs.

3. **Model Evaluation**:
   - Generated classification reports and confusion matrices.
   - Visualized ROC and Precision-Recall curves.
   - Provided a bar plot for performance metrics (Precision, Recall, F1 Score).

## **Dependencies**
- Python 3.7 or later
- Libraries:
  - `transformers`
  - `datasets`
  - `torch`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

## Results

### Evaluation Summary:

- F1 Score: 0.5061
- Accuracy: 59.40%
- AUC-ROC: 0.6032

