# Japanese Bar Exam LLM: Fine-Tuning & Self-Verification

This repository contains the official code and experimental scripts for our paper: **[Self-Verification is All You Need to Pass the Japanese Bar Examination](http://arxiv.org/abs/2601.03144)**.

We present a method to achieve passing scores on the Japanese Bar Examination (Short-Answer Test / 短答式) using a fine-tuned GPT-4.1 model with a specialized self-verification mechanism, without altering the authentic question format.

## Dataset

We utilize our newly constructed dataset, which faithfully replicates the original exam format (including complex multi-proposition constraints).

* **Hugging Face:** [`shinysup/JBE-MC-original-format`](https://huggingface.co/datasets/shinysup/JBE-MC-original-format)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shinandrew/self_verification.git
    cd self_verification
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set your OpenAI API Key:**
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

## Usage

### 1. Fine-Tuning (`finetune.py`)

This script automates the entire fine-tuning pipeline. It performs the following steps:
1.  Downloads our dataset (`shinysup/JBE-MC-original-format`) from Hugging Face.
2.  Formats the training data (years R1–R5) into the OpenAI JSONL format.
3.  Uploads the file to OpenAI.
4.  Initiates a supervised fine-tuning job on `gpt-4.1-2025-04-14`.
5.  Monitors the job status until completion and outputs the final model ID.

```bash
python finetune.py
```

Note: You will need the resulting Model ID (e.g., ft:gpt-4.1...) to update the configuration in run_experiments.py.

### 2. Running Experiments (`run_experiments.py`)
This script evaluates various models and methods on the Reiwa 6 (R6) exam data (the test set). It combines the train and test splits from the Hugging Face dataset to ensure correct year-based filtering.

The script runs the following experimental baselines:

- Zero-Shot: Base model directly answering the question.

- Few-Shot: Base model with 5 examples.

- Base + Self-Verify: Base model with our verification prompt.

- Fine-Tuned: The model trained via finetune.py.

- Self-Verification (Ours): The fine-tuned model combined with our two-step inference and verification logic.

Configuration: Open run_experiments.py and update the MODEL_FINETUNED variable with your specific model ID obtained from step 1.

```Python
# run_experiments.py
MODEL_FINETUNED = "ft:gpt-4.1-2025-04-14:personal::YourModelID"
```

Run the evaluation:

```bash
python run_experiments.py
```
Results will be saved to exam_predictions_merged.jsonl, containing detailed logs of inputs, outputs, canonicalized answers, and exact scoring breakdown.


## Citation
TBA
