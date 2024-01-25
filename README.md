# Transformer Emotion Analysis

## Overview

This project demonstrates the use of a Transformer architecture for emotion analysis. The Transformer model is fine-tuned on an emotion dataset, and the hidden states are extracted for downstream tasks such as logistic regression and evaluation using a dummy classifier.

## Getting Started

### Prerequisites

- Python 3.6 or later
- Install required packages: `pip install -r requirements.txt`

### Example Usage

```python
from datasets import list_datasets
from datasets import load_dataset
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# ... (rest of your code)

y_preds = lr_clf.predict(X_valid)
# plot_confusion_matrix(y_preds, y_valid, labels)
print(y_preds)
