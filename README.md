# LLM-Optimization
Contains code and resources for optimizing large language models using multi-degree low-rank approximations. Includes Python scripts for model fine-tuning with Hugging Face Transformers, low-rank approximation implementation, performance comparisons (perplexity, BLEU score), and instructions for various hardware setups.

# (Low-Rank Fine-Tuning for Sequence-to-Sequence Language Models)[BART_Tokenizer_optimizaiton.py]
This script implements baseline and low-rank approximation fine-tuning of a BART model for sequence-to-sequence tasks using the wmt16 dataset for English-to-Romanian translation. The script measures the effects of low-rank approximation on BLEU score, perplexity, memory usage, and training time, comparing different ranks.
