import os
import json
import time
import random
import pandas as pd
from openai import OpenAI

SAMPLE_SIZE = 100
OUTPUT_FILE = "outputs.csv"
RANDOM_SEED = 42

# Ollama runs locally — no API key needed
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODELS = [
    "llama3.1:8b",
    "qwen2.5:7b",
    "phi3:mini",
]

def load_prompts(path, n):
    """Load n prompts from the JSONL, picking a mix of challenging and normal."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))

    challenging = [p for p in prompts if p.get("challenging")] #Collects all prompts that are marked as challenging True
    normal = [p for p in prompts if not p.get("challenging")] #Collects all prompts that are marked as challenging False

    random.seed(RANDOM_SEED)
    random.shuffle(challenging)
    random.shuffle(normal)

    half = n // 2
    sampled = challenging[:half] + normal[:half]
    return sampled[:n]

def dominant_category(prompt_scores):
    """Return the toxicity category with the highest score for this prompt."""
    categories = ["toxicity", "profanity", "identity_attack", "threat", "insult", "sexually_explicit"]
    return max(categories, key=lambda c: prompt_scores.get(c, 0))

def query_model(model, prompt_text):
    """Send a prompt to a model and return its text response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ERROR with {model}: {e}")
        return None

def main():
    prompts = load_prompts("prompts.jsonl", SAMPLE_SIZE)
    print(f"Loaded {len(prompts)} prompts\n")

    rows = []
    for i, entry in enumerate(prompts):
        prompt_text = entry["prompt"]["text"]
        prompt_scores = entry["prompt"]
        category = dominant_category(prompt_scores)

        print(f"[{i+1}/{SAMPLE_SIZE}] Category: {category}")
        print(f"  Prompt: {prompt_text[:80]}...")

        for model in MODELS:
            short_name = model.split("/")[-1]
            print(f"  Querying {short_name}...", end=" ", flush=True)
            response = query_model(model, prompt_text)
            print("done" if response else "failed")

            rows.append({
                "prompt_text": prompt_text,
                "prompt_toxicity": prompt_scores.get("toxicity"),
                "prompt_category": category,
                "model": short_name,
                "response_text": response,
            })

            time.sleep(1)  # avoid rate limiting

        print()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
