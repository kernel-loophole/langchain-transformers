from datasets import load_metric
from datasets import load_dataset
import pandas as pd
import numpy as np
dataset = load_dataset("cnn_dailymail", "3.0.0")
bleu_metric = load_metric("sacrebleu")
bleu_metric.add(prediction="the the the the the the", reference=["the cat is on the mat"])
results = bleu_metric.compute(smooth_method="floor", smooth_value=0)
results["precisions"] = [np.round(p, 2) for p in
results["precisions"]]
data=pd.DataFrame.from_dict(results, orient="index", columns=["Value"])
bleu_metric.add(prediction="the cat is on mat", reference=["the cat is on the mat"])
results = bleu_metric.compute(smooth_method="floor", smooth_value=0)
results["precisions"] = [np.round(p, 2) for p in
results["precisions"]]
pd.DataFrame.from_dict(results, orient="index", columns=["Value"])
# def rouge_mat():
#     rouge_metric = load_metric("rouge")
#     reference = dataset["train"][1]["highlights"]
#     records = []
#     rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
#     for model_name in summaries:
#         rouge_metric.add(prediction=summaries[model_name],
#         reference=reference)
#         score = rouge_metric.compute()
#         rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in
#         rouge_names)
#         records.append(rouge_dict)
#     pd.DataFrame.from_records(records, index=summaries.keys())
print(data)