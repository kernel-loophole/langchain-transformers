from transformers import AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForQuestionAnswering
import torch
model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
question = "How much music can this hold?"
context = """An MP3 is about 1 MB/minute, so about 6000 hours depending on \ file size."""
inputs = tokenizer(question, context, return_tensors="pt")
print(tokenizer.decode(inputs["input_ids"][0]))
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(f"Input IDs shape: {inputs.input_ids.size()}")
print(f"Start logits shape: {start_logits.size()}")
print(f"End logits shape: {end_logits.size()}")
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits) + 1
answer_span = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_span)
print(f"Question: {question}")
print(f"Answer: {answer}")
pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
pipe(question=question, context=context, top_k=3)
pipe(question="Why is there no data?", context=context,
handle_impossible_answer=True)
def train_data(dfs):
    example = dfs["train"].iloc[0][["question", "context"]]
    tokenized_example = tokenizer(example["question"], example["context"],
    return_overflowing_tokens=True, max_length=100,
    stride=25)
    for idx, window in enumerate(tokenized_example["input_ids"]):
        print(f"Window #{idx} has {len(window)} tokens")