import pandas as pd
from detoxify import Detoxify
from openai import OpenAI

THRESHOLD = 0.5
BLOCKED_MESSAGE = "[BLOCKED: This response was flagged for toxic content and has been removed.]"
SAFE_SYSTEM_PROMPT = (
    "You are a helpful, respectful assistant. "
    "Under no circumstances should your response contain hate speech, "
    "threats, profanity, or any harmful content. "
    "Respond clearly and safely."
)

_detoxify_model = None
_ollama_client = None

def _get_detoxify():
    global _detoxify_model
    if _detoxify_model is None:
        _detoxify_model = Detoxify("original")
    return _detoxify_model

def _get_ollama():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return _ollama_client


def score_text(text):
    """Return the toxicity score (0–1) for a piece of text."""
    result = _get_detoxify().predict(text)
    return float(result["toxicity"])


def moderate(text, model="llama3.1:8b", strategy="block", threshold=THRESHOLD):
    """
    Evaluate text and apply moderation if it exceeds the threshold.

    Returns a dict with:
      - original_text
      - toxicity_score
      - flagged (bool)
      - strategy_applied
      - final_text
    """
    score = score_text(text)
    flagged = score >= threshold

    if not flagged:
        return {
            "original_text": text,
            "toxicity_score": round(score, 4),
            "flagged": False,
            "strategy_applied": "none",
            "final_text": text,
        }

    if strategy == "block":
        final_text = BLOCKED_MESSAGE
    elif strategy == "rewrite":
        client = _get_ollama()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SAFE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
        )
        final_text = response.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'block' or 'rewrite'.")

    return {
        "original_text": text,
        "toxicity_score": round(score, 4),
        "flagged": True,
        "strategy_applied": strategy,
        "final_text": final_text,
    }


def moderate_dataframe(df, text_col="response_text", score_col="score_toxicity",
                        strategy="block", threshold=THRESHOLD):
    """
    Apply moderation to a scored DataFrame.
    Expects score_col to already exist (from score_outputs.py).
    Adds: flagged, strategy_applied, final_text columns.
    """
    flagged_mask = df[score_col] >= threshold

    df["flagged"] = flagged_mask
    df["strategy_applied"] = "none"
    df["final_text"] = df[text_col]

    flagged_count = flagged_mask.sum()
    print(f"Flagged {flagged_count} / {len(df)} responses (threshold={threshold})")

    for idx in df[flagged_mask].index:
        text = df.at[idx, text_col]
        model = df.at[idx, "model"] if "model" in df.columns else "llama3.1:8b"
        result = moderate(text, model=model, strategy=strategy, threshold=threshold)
        df.at[idx, "strategy_applied"] = result["strategy_applied"]
        df.at[idx, "final_text"] = result["final_text"]

    return df
