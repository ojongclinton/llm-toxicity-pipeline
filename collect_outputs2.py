import os
import json
import time
import random
import pandas as pd
from openai import OpenAI

RANDOM_SEED = 42

MODELS = [
    "llama3.1:8b",
    "qwen2.5:7b",
    "phi3:mini",
]

def collect_prompts(filename , amount):
    all_prompts=[]
    with open(filename ,encoding="utf-8") as f:
        for line in f:
            all_prompts.append(json.loads(line))
            
    challenging = [p for p in all_prompts if p.get("challenging")] #Collects all prompts that are marked as challenging True
    normal = [p for p in all_prompts if not p.get("challenging")] #Collects all prompts that are marked as challenging False    

def main():
    print("Starting output collection...")
    prompts = collect_prompts("prompts.jsonl", 100)
    
    
if __name__ == "__main__":
    main()