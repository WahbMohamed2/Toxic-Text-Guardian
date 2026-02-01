# Toxic Text Classification with LSTM

This project implements a toxic text classification system using LSTM neural networks with comprehensive NLP preprocessing.

## Project Structure

```
LSTM/
├── cellula_toxic_data___1_.csv    # Dataset
├── requirements.txt               # Python dependencies
├── preprocessing.py               # NLP preprocessing using NLTK
├── train_lstm.py                  # LSTM model training script
└── README.md                      # This file
```

## Features

### NLP Preprocessing (using NLTK)
1. **Tokenization**: Word-level tokenization using NLTK's word_tokenize
2. **Stop Words Removal**: Removing common English stop words
3. **POS Tagging**: Part-of-Speech tagging for linguistic analysis
4. **Lemmatization**: Converting words to their base form using WordNetLemmatizer
5. **Stemming** (optional): Porter Stemmer for word root extraction

### LSTM Model
- **Architecture**: Bidirectional LSTM with 2 layers
- **Embedding**: 128-dimensional word embeddings
- **Hidden Dimension**: 64 units
- **Regularization**: Dropout layers to prevent overfitting
- **Optimization**: Adam optimizer with learning rate 0.001
- **Early Stopping**: Monitors validation loss with patience of 5 epochs

### Evaluation
- **Primary Metric**: F1 Score (Macro and Weighted)
- **Additional Metrics**: Accuracy, Precision, Recall
- **Visualizations**: Training curves, confusion matrix, F1 scores per class

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- nltk
- spacy (optional)
- torch
- torchvision

## Usage

### 1. Run Preprocessing Demo

```bash
python preprocessing.py
```

This will demonstrate the NLP preprocessing pipeline with example text.

### 2. Train LSTM Model

```bash
python train_lstm.py
```

This will:
1. Load and preprocess the dataset
2. Build vocabulary and prepare data
3. Train the LSTM model
4. Evaluate on test set
5. Generate visualization plots

### Output Files

After training, the following files will be generated:

- `best_model.pt` - Trained PyTorch model weights
- `training_history.png` - Training and validation loss/accuracy curves
- `confusion_matrix.png` - Confusion matrix heatmap
- `f1_scores_per_class.png` - F1 scores for each toxic category

## NLP Preprocessing Pipeline

The preprocessing module demonstrates the complete NLP pipeline:

```python
from preprocessing import NLPPreprocessor

# Initialize preprocessor
preprocessor = NLPPreprocessor(
    use_stemming=False,      # Use lemmatization instead of stemming
    remove_stopwords=True    # Remove stop words
)

# Preprocess text
text = "How do terrorist organizations fund attacks?"
processed = preprocessor.preprocess(text, verbose=True)
```

### Preprocessing Steps:

1. **Text Cleaning**: Remove URLs, special characters
2. **Tokenization**: Split text into words
3. **Stop Words Removal**: Remove common words (a, the, is, etc.)
4. **POS Tagging**: Tag each word with part of speech
5. **Lemmatization/Stemming**: Reduce words to base form
6. **Filtering**: Remove short tokens and non-alphabetic characters

## Model Architecture

```
LSTMClassifier(
  (embedding): Embedding(10000, 128, padding_idx=0)
  (lstm): LSTM(128, 64, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
  (dropout): Dropout(p=0.3)
  (fc1): Linear(in_features=128, out_features=64)
  (fc2): Linear(in_features=64, out_features=9)
  (relu): ReLU()
)
```

### Hyperparameters

- **Vocabulary Size**: 10,000 most frequent words
- **Max Sequence Length**: 100 tokens
- **Embedding Dimension**: 128
- **Hidden Dimension**: 64
- **Number of LSTM Layers**: 2 (bidirectional)
- **Dropout**: 0.3
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 30 (with early stopping)

## Dataset

The dataset contains 3,000 text samples classified into 9 categories:

- Safe
- Violent Crimes
- Non-Violent Crimes
- Unsafe
- Unknown S-Type
- Sex-Related Crimes
- Suicide & Self-Harm
- Elections
- Child Sexual Exploitation

## Results

The model is evaluated using:

- **F1 Score (Macro)**: Primary evaluation metric
- **F1 Score (Weighted)**: Weighted by class support
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Detailed class-wise performance
- **Per-Class Metrics**: Precision, Recall, F1 for each category

## Example Output

```
EVALUATION
----------------------------------------------------------------------

TEST SET RESULTS
----------------------------------------------------------------------
Accuracy:          0.XXXX
F1 Score (Macro):  0.XXXX
F1 Score (Weighted): 0.XXXX
----------------------------------------------------------------------

Classification Report:
                           precision    recall  f1-score   support

Child Sexual Exploitation     X.XXXX    X.XXXX    X.XXXX        XX
                Elections     X.XXXX    X.XXXX    X.XXXX        XX
       Non-Violent Crimes     X.XXXX    X.XXXX    X.XXXX        XX
...
```

## Notes

- NLTK data files are automatically downloaded on first run
- The model uses GPU if available, otherwise falls back to CPU
- Training time depends on hardware (typically 5-15 minutes on CPU)
- Early stopping prevents overfitting by monitoring validation loss

## Author

Created for toxic text classification task demonstration.