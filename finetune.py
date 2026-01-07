import json
import time
import sys
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset

# =========================
# CONFIGURATION
# =========================

# Data Config
OUT_DIR = Path("./finetune_data")
OUT_DIR.mkdir(exist_ok=True)
TRAIN_JSONL = OUT_DIR / "lawexam_R1_R5_train.jsonl"
SYSTEM_PROMPT = "You are a Japanese law exam solver. Answer concisely using only symbols or numbers."

# Fine-tuning Config
BASE_MODEL = "gpt-4.1-2025-04-14"
POLL_INTERVAL = 10  # seconds

# =========================
# STEP 1: PREPARE DATA
# =========================

print("📥 Loading dataset from Hugging Face...")

# Load the dataset from Hugging Face
dataset = load_dataset("shinysup/JBE-MC-original-format")
train_data = dataset["train"]

print(f"Training samples: {len(train_data)}")

# Export to JSONL format for OpenAI
print(f"💾 Saving training data to {TRAIN_JSONL}...")
with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
    for r in train_data:
        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(r["question"])},
                {"role": "assistant", "content": str(r["answer"])}
            ]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Data preparation complete.")

# =========================
# STEP 2: FINE-TUNE
# =========================

client = OpenAI()

if not TRAIN_JSONL.exists():
    print(f"❌ Training file not found: {TRAIN_JSONL}")
    sys.exit(1)

# Upload Training File
print("\n📤 Uploading training file to OpenAI...")
with open(TRAIN_JSONL, "rb") as f:
    upload = client.files.create(
        file=f,
        purpose="fine-tune"
    )

training_file_id = upload.id
print(f"✅ File uploaded successfully. ID: {training_file_id}")

# Create Fine-Tuning Job
print(f"🚀 Starting fine-tuning job for model: {BASE_MODEL}...")

job = client.fine_tuning.jobs.create(
    model=BASE_MODEL,
    training_file=training_file_id,
    seed=820456509
)

job_id = job.id
print(f"🧠 Fine-tuning job created. Job ID: {job_id}")

# Monitor Job Status
print("\n📡 Monitoring fine-tuning progress...\n")

while True:
    job = client.fine_tuning.jobs.retrieve(job_id)
    status = job.status

    print(f"[STATUS] {status}")

    if status == "succeeded":
        print("\n🎉 Fine-tuning completed successfully!")
        print("🔗 Fine-tuned model name:")
        print(f"    {job.fine_tuned_model}")
        break

    if status in ("failed", "cancelled"):
        print("\n❌ Fine-tuning did not complete successfully.")
        if hasattr(job, "error") and job.error:
            print("Error details:")
            print(job.error)
        break

    time.sleep(POLL_INTERVAL)

print("\n✅ Script finished.")
