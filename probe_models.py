import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
client = InferenceClient(token=os.getenv("HF_API_KEY"))

CANDIDATES = [
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]

TEST_MSG = [{"role": "user", "content": "Say the word OK and nothing else."}]

for model in CANDIDATES:
    try:
        print("Starting testttt")
        r = client.chat_completion(model=model, messages=TEST_MSG, max_tokens=10)
        print(f"  PASS  {model}")
    except Exception as e:
        short = str(e)[:120]
        print(f"  FAIL  {model}  →  {short}")
