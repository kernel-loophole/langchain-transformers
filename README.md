# Transformer Emotion Analysis

## Overview

This project demonstrates the use of a Transformer architecture for emotion analysis. The Transformer model is fine-tuned on an emotion dataset, and the hidden states are extracted for downstream tasks such as logistic regression and evaluation using a dummy classifier.


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
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# all_datasets = list_datasets()
# print(f"There are {len(all_datasets)} datasets currently available on the Hub")
# print(f"The first 10 are: {all_datasets[:10]}")
emotions = load_dataset("emotion")
# print(emotions)
train_ds = emotions["train"]
# print(train_ds)
# print(len(train_ds))
# print(train_ds[0])
# print(train_ds[1])
# print(train_ds.features)
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
print(emotions["train"].column_names)
print(tokenize(emotions["train"][:2]))
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded["train"].column_names)
# emotions.set_format(type="pandas")
# df = emotions["train"][:]
# df["label_name"] = df["label"].apply(label_int2str)
# # print(df.head())
# # df["label_name"].value_counts(ascending=True).plot.barh()
# # # plt.title("Frequency of Classes")
# # plt.show()
# df["Words Per Tweet"] = df["text"].str.split().apply(len)
# df.boxplot("Words Per Tweet", by="label_name", grid=False,
# showfliers=False, color="black")
# emotions.reset_format()
def extract_hidden_states(batch):
# Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items()
    if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
     # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
from transformers import AutoModel
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
# print(f"Input tensor shape: {inputs['input_ids'].size()}")
inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
emotions_encoded.set_format("torch",
columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
import numpy as np
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
# print(X_train.shape, X_valid.shape)
X_scaled = MinMaxScaler().fit_transform(X_train)
# Initialize and fit UMAP
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_clf.score(X_valid, y_valid)
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)
def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
y_preds = lr_clf.predict(X_valid)
# plot_confusion_matrix(y_preds, y_valid, labels)
print(y_preds)
