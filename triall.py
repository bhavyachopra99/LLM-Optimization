from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate
from datasets import load_dataset
from evaluate import load  
import pandas as pd
import gc
from accelerate import Accelerator

# Initialize Accelerator
accelerator = Accelerator()

# Set device to MPS (Metal Performance Shaders)
device = torch.device("mps")

# Load the tokenizer and model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load BLEU and ROUGE metrics from the evaluate library
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# Low-Rank Approximation Function
def low_rank_approximation(weight_matrix, rank):
    u, s, v = torch.svd(weight_matrix)
    s[rank:] = 0
    low_rank_weight = torch.mm(u, torch.mm(torch.diag(s), v.t()))
    return low_rank_weight

# Apply Low-Rank Approximation to Model
def apply_low_rank_approximation(model, rank=50):
    layer = model.transformer.h[0].attn.c_attn
    original_weight = layer.weight
    approx_weight = low_rank_approximation(original_weight, rank)
    layer.weight = torch.nn.Parameter(approx_weight)
    return model

# Load and prepare datasets
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

train_val_test_split = dataset.train_test_split(test_size=0.4, seed=42)
train_dataset = train_val_test_split['train']
val_test_split = train_val_test_split['test'].train_test_split(test_size=0.5, seed=42)
val_dataset = val_test_split['train']
test_dataset = val_test_split['test']

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Data collator to include labels
def data_collator(batch):
    input_ids = torch.stack([example['input_ids'] for example in batch])
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Training arguments setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=False,
    gradient_accumulation_steps=2,
)

# Fine-tuning function
def fine_tune_model(model, model_name, rank=None):
    if rank:
        model = apply_low_rank_approximation(model, rank)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    model.save_pretrained(f"./{model_name}_fine_tuned_rank_{rank}")
    tokenizer.save_pretrained(f"./{model_name}_fine_tuned_rank_{rank}")
    del trainer
    gc.collect()
# Model evaluation function with BLEU and ROUGE scores
def evaluate_model(model_name, rank=None):
    # Reload the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(f"./{model_name}_fine_tuned_rank_{rank}").to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Evaluate the Model
    with torch.no_grad():  # Disable gradient calculation for evaluation
        predictions = trainer.predict(tokenized_val_dataset)

    # Measure Inference Time
    import time
    start_time = time.time()
    with torch.no_grad():  # Disable gradient calculation for inference
        predictions_test = trainer.predict(tokenized_test_dataset)
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Convert predicted logits to text for BLEU/ROUGE
    pred_logits = torch.tensor(predictions.predictions)  # Convert to tensor if it's a NumPy array
    pred_texts = tokenizer.batch_decode(torch.argmax(pred_logits, dim=-1), skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

    # Prepare references for BLEU
    references = [[label_text] for label_text in label_texts]  # Wrap each label in a list

    # Calculate BLEU score
    bleu_score = bleu_metric.compute(predictions=pred_texts, references=references)

    # Calculate ROUGE score
    rouge_score = rouge_metric.compute(predictions=pred_texts, references=references)

    # Debugging print statements
    print(f"Evaluation Results for {model_name}: {predictions.metrics}")
    print(f"Inference Time for {model_name}: {inference_time} seconds")

    # Clear memory after evaluation
    del model, trainer
    gc.collect()

    return predictions.metrics, inference_time, bleu_score, rouge_score

# Function to compare results across ranks and display in a table
# Function to compare results across ranks and display in a table
def compare_models(model_name, ranks):
    # Prepare a dataframe to store results
    columns = ['Rank', 'Eval_Loss', 'Eval_Runtime', 'Inference_Time', 'BLEU', 'ROUGE_L']
    results_df = pd.DataFrame(columns=columns)
    
    for rank in ranks:
        print(f"Fine-tuning and evaluating for rank: {rank}")
        
        # Load base model for fine-tuning
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Fine-tune the model for the specific rank
        fine_tune_model(model, model_name, rank)
        
        # Evaluate the model
        results, inference_time, bleu_score, rouge_score = evaluate_model(model_name, rank)
        
        # Check if the results contain expected keys
        eval_loss = results.get('eval_loss', None)
        eval_runtime = results.get('eval_runtime', None)

        # Create a new DataFrame for the current results
        new_data = pd.DataFrame({
            'Rank': [rank],
            'Eval_Loss': [eval_loss],
            'Eval_Runtime': [eval_runtime],
            'Inference_Time': [inference_time],
            'BLEU': [bleu_score['bleu']],
            'ROUGE_L': [rouge_score['rougeL'] if 'rougeL' in rouge_score else None]  # Updated here
        })
        
        # Concatenate the new data to the existing results DataFrame
        results_df = pd.concat([results_df, new_data], ignore_index=True)
        
        # Garbage collection
        del model
        gc.collect()
    
    # Display results in a table
    print(results_df)
    return results_df

# Run the comparison for ranks 0, 50, 100, 150, 200
ranks = [0, 50, 100, 150, 200]
results_table = compare_models(model_name, ranks)

# Save results to CSV
results_table.to_csv("fine_tuning_results.csv", index=False)
# Run the comparison for ranks 0, 50, 100, 150, 200
ranks = [0, 50, 100, 150, 200]
results_table = compare_models(model_name, ranks)

# Save results to CSV
results_table.to_csv("fine_tuning_results.csv", index=False)