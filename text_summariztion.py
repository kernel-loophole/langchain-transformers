from datasets import load_dataset
import nltk
from transformers import pipeline, set_seed
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
#CNN Dailymail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
print(f"Features: {dataset['train'].column_names}")
sample = dataset["train"][1]
print(f"""
Article (excerpt of 500 characters, total length:
{len(sample["article"])}):
""")
print(sample["article"][:500])
print(f'\nSummary (length: {len(sample["highlights"])}):')
print(sample["highlights"])
sample_text = dataset["train"][1]["article"][:2000]
# We'll collect the generated summaries of each model in a
summaries = {}

string = "The U.S. are a country. The U.N. is an organization."
sent_tokenize(string)
def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])
summaries["baseline"] = three_sentence_summary(sample_text)
set_seed(42)
pipe = pipeline("text-generation", model="gpt2-xl")
gpt2_query = sample_text + "\nTL;DR:\n"
pipe_out = pipe(gpt2_query, max_length=512,
clean_up_tokenization_spaces=True)
summaries["gpt2"] = "\n".join(
sent_tokenize(pipe_out[0]["generated_text"][len(gpt2_query) :]))

def t5_model():
    pipe = pipeline("summarization", model="t5-large")
    pipe_out = pipe(sample_text)
    summaries["t5"] = "\n".join(sent_tokenize(pipe_out[0]
    ["summary_text"]))
    