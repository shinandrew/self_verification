import re
import time
import json
import random
import pandas as pd
from collections import defaultdict
from typing import Dict, List
from openai import OpenAI
from datasets import load_dataset

# =========================
# CONFIG
# =========================

HF_DATASET = "shinysup/JBE-MC-original-format"
MODEL_BASE = "gpt-4.1"
MODEL_FINETUNED = "ft:gpt-4.1-2025-04-14:personal::Your_Model_ID"
RESULTS_PATH = "exam_predictions_merged.jsonl" 

LOG_EVERY = 10
FEW_SHOT_K = 5
TEMPERATURE = 0.4

random.seed(42)

# =========================
# SHARED PROMPTS
# =========================

ANSWER_FORMAT_INSTRUCTION = """
【回答形式の厳守】
必ず「答えのみ」を出力せよ。理由・説明・記号は一切不要。

1) 選択肢が番号で与えられている場合
   （例：1. アO イO ウO、2. アO イO ウX …）
   → 正しい選択肢の番号のみ出力（例：2）

2) 各記述（ア・イ・ウ…）について 1 / 2 を答える問題の場合
   → 各記述の答えを順に連結した数字列のみ出力（例：112）

禁止：
- OOX
- アO イO ウX
- ア1 イ1 ウ2
- 説明文
"""

# =========================
# UNIFIED EXPERIMENT PROMPTS
# =========================

SYSTEM_PROMPT_UNIFIED = "あなたは日本の法律試験を解く受験者である。"

VERIFICATION_PROMPT = """
あなたは法律試験の答案を最終確認する役割である。

以下の【問題】と【あなたの解答】を照らし合わせ、
選択肢番号または数値の形式として
最も正しい最終解答を一つだけ出力せよ。

・問題文の条件に照らして明らかに誤っている場合のみ修正すること
・元の解答が正しい場合は、そのまま同じ解答を出力すること
・理由や説明は一切出力せず、最終的な数字のみを出力せよ
"""

# =========================
# CLIENT
# =========================

client = OpenAI()

# =========================
# MODEL CALLS
# =========================

def call_model(prompt: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Japanese law exam solver."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()

def call_model_unified(prompt: str):
    resp = client.chat.completions.create(
        model=MODEL_FINETUNED,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_UNIFIED},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE, 
    )
    return resp.choices[0].message.content.strip()

def build_few_shot_prompt(target_question, train_df, k=5):
    # Sample k random examples from training data
    examples = train_df.sample(k).to_dict("records")
    
    prompt = ANSWER_FORMAT_INSTRUCTION + "\n\n"
    
    # Append examples
    for ex in examples:
        prompt += f"問題：\n{ex['question']}\n\n答え：\n{ex['answer']}\n\n---\n\n"
        
    # Append the actual target question
    prompt += f"問題：\n{target_question}"
    return prompt

# =========================
# BASE VERIFICATION LOGIC
# =========================

def self_verify_answer(question: str, initial_answer: str, model: str) -> str:
    verify_prompt = f"""
{VERIFICATION_PROMPT}

【問題】
{question}

【あなたの解答】
{initial_answer}
"""
    return call_model(verify_prompt, model)

def base_self_verify_runner(r):
    initial = call_model(
        f"{ANSWER_FORMAT_INSTRUCTION}\n\n問題：\n{r.question}",
        MODEL_BASE
    )
    return self_verify_answer(r.question, initial, MODEL_BASE)

# =========================
# UNIFIED EXPERIMENT LOGIC
# =========================

def verify_answer_unified(question, draft_answer):
    prompt = f"""
{VERIFICATION_PROMPT}

【問題】
{question}

【あなたの解答】
{draft_answer}
"""
    verified = call_model_unified(prompt)
    canon = canonicalize_answer(verified)
    return "".join(canon), verified

def solve_question_unified(question):
    parts = []
    
    parts.append(ANSWER_FORMAT_INSTRUCTION)
    parts.append("問題：\n" + question)

    prompt = "\n\n".join(parts)
    raw_draft = call_model_unified(prompt)
    draft = "".join(canonicalize_answer(raw_draft))

    final, raw_verified = verify_answer_unified(question, draft)

    return final 

# =========================
# NORMALIZATION & SCORING
# =========================

def canonicalize_answer(ans):
    """
    Very strict canonicalization:
    - Extract digits only
    - '1'   -> ('1',)
    - '112' -> ('1','1','2')
    """
    if ans is None:
        return ()

    s = str(ans)
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    digits = re.findall(r"\d", s)
    if digits:
        return tuple(digits)
    return ()

def answers_match(pred, gold):
    return canonicalize_answer(pred) == canonicalize_answer(gold)

def points_for_pair(pred, gold, max_point):
    p = canonicalize_answer(pred)
    g = canonicalize_answer(gold)

    if not p or not g:
        return 0

    if p == g:
        return max_point

    if max_point >= 3 and len(g) >= 3 and len(p) == len(g):
        mismatches = sum(1 for x, y in zip(p, g) if x != y)
        if mismatches == 1:
            return max(max_point - 2, 0)

    return 0


# =========================
# EVALUATION LOOP
# =========================

def run_experiment(name, runner):
    correct = 0
    points = 0
    max_points = 0

    section_points = defaultdict(int)
    section_max = defaultdict(int)

    for i, r in test.iterrows():
        t0 = time.time()
        pred = runner(r)
        latency = time.time() - t0

        ok = answers_match(pred, r.answer)
        pt = points_for_pair(pred, r.answer, r.point)
        section_points[r.type] += pt
        section_max[r.type] += r.point

        record = {
            "variant": name,
            "question_id": int(i),
            "year": r.year,
            "type": r.type,
            "point": r.point,
            "question": r.question,
            "gold_answer": str(r.answer),
            "prediction": pred,
            "gold_canonical": "".join(canonicalize_answer(r.answer)),
            "pred_canonical": "".join(canonicalize_answer(pred)),
            "correct": bool(ok),
            "points_awarded": pt,
        }

        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


        #print("CANON:", canonicalize_answer(pred),
        #      "GOLD:", canonicalize_answer(r.answer))

        correct += int(ok)
        points += pt
        max_points += r.point

        if i == 0 or (i + 1) % LOG_EVERY == 0:
            print(
                f"[{name}] {i+1}/{len(test)} "
                f"lat={latency:.2f}s "
                f"pred={pred} gold={r.answer} "
                f"acc={correct/(i+1):.3f}"
            )

    return {
        "accuracy": correct / len(test),
        "point_ratio": points / max_points,
        "points_actual": points,
        "points_max": max_points,
        "points_by_section": dict(section_points),
        "points_by_section_max": dict(section_max),
    }


# =========================
# MAIN
# =========================

dataset = load_dataset(HF_DATASET)

# Combine 'train' and 'test' splits to recreate the full dataset
# allowing the existing filtering logic to work exactly as before
df = pd.concat([
    pd.DataFrame(dataset['train']),
    pd.DataFrame(dataset['test'])
]).reset_index(drop=True)

train = df[df.year != "R6"].reset_index(drop=True)
test = df[df.year == "R6"].reset_index(drop=True)

# Helper to wrap the unified solver into the runner format
def self_verification(r):
    return solve_question_unified(r.question)

experiments = {
    "zero_shot": lambda r: call_model(
        f"{ANSWER_FORMAT_INSTRUCTION}\n\n問題：\n{r.question}",
        MODEL_BASE
    ),
    "few_shot": lambda r: call_model(
        build_few_shot_prompt(r.question, train, k=FEW_SHOT_K), # Use the helper here
        MODEL_BASE
    ),
    "base_self_verify": base_self_verify_runner,
    "finetuned": lambda r: call_model(
        f"{ANSWER_FORMAT_INSTRUCTION}\n\n問題：\n{r.question}",
        MODEL_FINETUNED
    ),
    
    "self_verification": self_verification,
}

for name, runner in experiments.items():
    print(f"\n=== RUNNING {name.upper()} ===")
    result = run_experiment(name, runner)
    print(json.dumps(result, indent=2, ensure_ascii=False))
