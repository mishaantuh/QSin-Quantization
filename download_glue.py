import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import sys 

model_checkpoint = "bert-base-uncased"
task = sys.argv[1]

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task, cache_dir='cur_metric/')
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

tokenizer.save_pretrained("./models/tokenizer-bert-base/")
model.save_pretrained("./models/model-bert-base/")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True,  padding=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.save_to_disk("cur_glue_data/")

def preprocess_function_st(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True,  padding='max_length')
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length')

encoded_dataset = dataset.map(preprocess_function_st, batched=True)
encoded_dataset.save_to_disk("cur_glue_data_st/")


