# CSE 151B SP26 Competition

## Overview
This solution uses **Qwen/Qwen3-4B-Thinking-2507** to solve math reasoning problems. For each question the model produces a full
step-by-step chain of thought ending with the final answer in `\boxed{}`, which the grader extracts.

## Hardware & Runtime
- **GPU used:** NVIDIA A30 (24 GB) / H100 MIG slice on UCSD DSMLP.
- **Approximate inference time:** ~1 minute per question at 4096 tokens; roughly
  12-16 hours for the full 943-question private set on a single GPU.

## Setup
The model weights download automatically from the Hugging Face Hub on first run
(`Qwen/Qwen3-4B-Thinking-2507`, ~8 GB).

```bash
pip install -r requirements.txt
```

## Running Inference
Single entry point (`run_inference()` in `run_inference.py`):

```bash
python run_inference.py \
    --private-path data/private.jsonl \
    --output-csv submission.csv
```
