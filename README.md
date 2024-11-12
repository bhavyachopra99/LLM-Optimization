# LLM-Optimization
This repository contains code and resources for optimizing large language models (LLMs) using multi-degree low-rank approximations. The goal is to explore and enhance model efficiency without compromising performance. The code is implemented in Python, utilizing the Hugging Face Transformers library for model fine-tuning.

## Contents
Model Fine-Tuning Scripts: Python scripts to fine-tune models with both baseline and low-rank approximations.
Low-Rank Approximation: Code for applying low-rank approximations to model layers, aimed at reducing computational requirements.
Performance Metrics: Evaluation metrics, including perplexity and BLEU score, for comparing baseline and low-rank models.

## Scripts
[Low-Rank Fine-Tuning for Sequence-to-Sequence Language Models](BART_Tokenizer_optimizaiton.py)
This script demonstrates both baseline and low-rank fine-tuning of a BART model on the wmt16 dataset for English-to-Romanian translation tasks. Key metrics measured include BLEU score, perplexity, memory usage, and training time, with comparisons across different ranks in low-rank approximations.

[Optimied fine tuning of LLMS with Ollama Containers](ollama.ipynb)

[Using distillgpt model with metal and optimizing it by using different ranks](mac_distillgpt_distillgpt.ipynb)

[Optimizing text generation](low_rank_optim_wikitext_textgen.ipynb)