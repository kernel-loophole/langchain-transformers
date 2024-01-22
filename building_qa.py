from datasets import load_dataset
import pandas as pd
import matplotlib as plt
from model_qa import train_data
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForQuestionAnswering

model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
subjqa = load_dataset("subjqa", name="electronics")
dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}
for split, df in dfs.items():
    print(f"Number of questions in {split}: {df['id'].nunique()}")
qa_cols = ["title", "question", "answers.text",
"answers.answer_start", "context"]
def get_question():
    counts = {}
    question_types = ["What", "How", "Is", "Does", "Do", "Was", "Where", "Why"]
    for q in question_types:
        counts[q] = dfs["train"]["question"].str.startswith(q).value_counts()[True]
    pd.Series(counts).sort_values().plot.barh()
    plt.title("Frequency of Question Types")
    plt.show()
sample_df = dfs["train"][qa_cols].sample(2, random_state=7)
start_idx = sample_df["answers.answer_start"].iloc[0][0]
end_idx = start_idx + len(sample_df["answers.text"].iloc[0][0])
sample_df["context"].iloc[0][start_idx:end_idx]
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
# with torch.no_grad():
#     outputs = model(**inputs)
#     print(outputs)
# if __name__=="__main__":
    # get_question()
train_data(dfs)