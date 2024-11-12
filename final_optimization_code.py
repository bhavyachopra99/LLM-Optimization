import time
import tracemalloc
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load dataset (e.g., 'wmt16' for translation)
dataset = load_dataset('wmt16', 'ro-en')

# Load tokenizer for the BART model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# Tokenization function
def tokenize_function(examples):
    source_texts = [example['en'] for example in examples['translation']]
    return tokenizer(source_texts, truncation=True, padding="max_length", max_length=128)

print("Tokenizing - \n")
# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

def preprocess_function(examples):
    source_texts = [example['en'] for example in examples['translation']]
    target_texts = [example['ro'] for example in examples['translation']]
    
    model_inputs = tokenizer(source_texts, truncation=True, padding="max_length", max_length=128)
    
    # Tokenize Romanian texts as labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_texts, truncation=True, padding="max_length", max_length=128)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Take 10% of both train and validation datasets
sample_size_train = int(len(tokenized_datasets['train']) * 0.1)
sample_size_eval = int(len(tokenized_datasets['validation']) * 0.1)

tokenized_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(sample_size_train)).map(preprocess_function, batched=True)
tokenized_eval_dataset = tokenized_datasets['validation'].shuffle(seed=42).select(range(sample_size_eval)).map(preprocess_function, batched=True)

# Function to evaluate model with BLEU score
def evaluate_model(trainer, eval_dataset):
    # Generate predictions
    predictions = trainer.predict(eval_dataset).predictions
    logits = predictions[0]

    # Convert logits to predicted token IDs
    predicted_ids = torch.argmax(F.softmax(torch.tensor(logits), dim=-1), dim=-1).tolist()

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

    # Extract references from the evaluation dataset
    decoded_refs = tokenizer.batch_decode(eval_dataset['labels'], skip_special_tokens=True)

    # Initialize BLEU score accumulator
    total_bleu_score = 0
    num_sentences = len(decoded_preds)

    # Smoothing function
    smoothing_function = SmoothingFunction()

    for pred, ref in zip(decoded_preds, decoded_refs):
        hypothesis = pred.split()
        reference = ref.split()

        # Compute BLEU score with smoothing
        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=smoothing_function.method1)
        total_bleu_score += bleu_score

    average_bleu_score = total_bleu_score / num_sentences if num_sentences > 0 else 0
    return average_bleu_score

# Function to calculate perplexity
def perplexity_eval(trainer, eval_dataset):
    perplexity_metric = evaluate.load("perplexity", module_type="metric")
    predictions = trainer.predict(eval_dataset).predictions
    logits = predictions[0]
    predicted_ids = torch.argmax(F.softmax(torch.tensor(logits), dim=-1), dim=-1).tolist()
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    results = perplexity_metric.compute(model_id='gpt2', add_start_token=False, predictions=decoded_preds)
    return results

# Normal fine-tuning function
def normal_fine_tuning():
    tracemalloc.start()
    baseline_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

    training_args = TrainingArguments(
        output_dir="./results_baseline",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=baseline_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )

    start_training_time = time.time()
    trainer.train()
    end_training_time = time.time()

    bleu_score = evaluate_model(trainer, tokenized_eval_dataset)
    perplexity = perplexity_eval(trainer, tokenized_eval_dataset)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "bleu_score": bleu_score,
        "perplexity": perplexity['mean_perplexity'],
        "training_time": end_training_time - start_training_time,
        "memory_current": current / (1024 * 1024),
        "memory_peak": peak / (1024 * 1024),
    }

# Low-rank approximation function
def low_rank_approximation(layer, rank):
    weight_matrix = layer.weight.data.cpu().numpy()
    U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
    U_reduced = U[:, :rank]
    S_reduced = S[:rank]
    Vt_reduced = Vt[:rank, :]
    low_rank_matrix = np.dot(U_reduced, np.dot(np.diag(S_reduced), Vt_reduced))
    layer.weight.data = torch.tensor(low_rank_matrix, device=layer.weight.device)

# Low-rank fine-tuning function
def low_rank_fine_tuning(rank=10):
    tracemalloc.start()
    low_rank_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

    # Apply low-rank approximation
    for name, layer in low_rank_model.named_modules():
        if isinstance(layer, nn.Linear):
            low_rank_approximation(layer, rank)

    training_args = TrainingArguments(
        output_dir="./results_low_rank",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=low_rank_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )

    start_training_time = time.time()
    trainer.train()
    end_training_time = time.time()

    bleu_score = evaluate_model(trainer, tokenized_eval_dataset)
    perplexity = perplexity_eval(trainer, tokenized_eval_dataset)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "bleu_score": bleu_score,
        "perplexity": perplexity['mean_perplexity'],
        "training_time": end_training_time - start_training_time,
        "memory_current": current / (1024 * 1024),
        "memory_peak": peak / (1024 * 1024),
    }

# Ranks for experimentation
ranks = [50, 100, 150, 200]

# List to store all results
results_list = []

# Run normal fine-tuning and add results to the list
print("Running baseline fine-tuning...")
baseline_metrics = normal_fine_tuning()
results_list.append({
    'Rank': 'Baseline',
    'BLEU Score': baseline_metrics['bleu_score'],
    'Perplexity': baseline_metrics['perplexity'],
    'Training Time (s)': baseline_metrics['training_time'],
    'Current Memory Usage (MB)': baseline_metrics['memory_current'],
    'Peak Memory Usage (MB)': baseline_metrics['memory_peak']
})

# Run low-rank fine-tuning for each rank and store the results
for rank in ranks:
    print(f"Running low-rank fine-tuning for rank {rank}...")
    low_rank_metrics = low_rank_fine_tuning(rank=rank)
    results_list.append({
        'Rank': rank,
        'BLEU Score': low_rank_metrics['bleu_score'],
        'Perplexity': low_rank_metrics['perplexity'],
        'Training Time (s)': low_rank_metrics['training_time'],
        'Current Memory Usage (MB)': low_rank_metrics['memory_current'],
        'Peak Memory Usage (MB)': low_rank_metrics['memory_peak']
    })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results_list)

# Export the DataFrame to a CSV file
results_df.to_csv('low_rank_fine_tuning_metrics.csv', index=False)

print("Metrics exported to low_rank_fine_tuning_metrics.csv")