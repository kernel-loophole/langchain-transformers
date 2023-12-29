import torch
import torch.nn.functional as F


text="this is simple text"
tokenized_text = list(text)
print(tokenized_text)
#ass
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
input_ids = [token2idx[token] for token in tokenized_text]
# print(token2idx)
print(input_ids)
#one hot encoding
input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
# print(one_hot_encodings.shape)
print(f"Token: {tokenized_text[0]}")
print(f"Tensor index: {input_ids[0]}")
print(f"One-hot: {one_hot_encodings[0]}")