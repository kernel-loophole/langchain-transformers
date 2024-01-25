from transformers import pipeline
from datasets import load_dataset
from datasets import load_metric
bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
pipe = pipeline("text-classification", model=bert_ckpt)
query = """Hey, I'd like to rent a vehicle from Nov 1st to Nov 15th in
Paris and I need a 15 passenger van"""
print(pipe(query))
class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="BERTbaseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type
    def compute_accuracy(self):
        """This overrides the PerformanceBenchmark.compute_accuracy()
        method"""
        preds, labels = [], []
        for example in self.dataset:
            pred = self.pipeline(example["text"])[0]["label"]
            label = example["intent"]
            preds.append(intents.str2int(pred))
            labels.append(label)
            accuracy = accuracy_score.compute(predictions=preds,
            references=labels)
        print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
        return accuracy
    def compute_size(self):
        pass
    def time_pipeline(self):
        pass
    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics
clinc = load_dataset("clinc_oos", "plus")
sample = clinc["test"][42]
intents = clinc["test"].features["intent"]
intents.int2str(sample["intent"])
accuracy_score = load_metric("accuracy")