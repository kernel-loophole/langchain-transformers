from datasets import list_datasets
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

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
emotions.set_format(type="pandas")
df = emotions["train"][:]
df["label_name"] = df["label"].apply(label_int2str)
# print(df.head())
# df["label_name"].value_counts(ascending=True).plot.barh()
# # plt.title("Frequency of Classes")
# plt.show()
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False,
showfliers=False, color="black")
emotions.reset_format()