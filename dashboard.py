import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from detoxify import Detoxify

st.set_page_config(page_title="Toxicity Pipeline Dashboard", layout="wide")

# ── Data loading ────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    return pd.read_csv("moderated_outputs.csv")

@st.cache_data
def load_baseline():
    """Pull continuation.toxicity from the dataset as the base-model baseline."""
    records = []
    with open("prompts.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            records.append({
                "prompt_category": max(
                    ["toxicity", "profanity", "identity_attack", "threat", "insult", "sexually_explicit"],
                    key=lambda c: float(entry["prompt"].get(c) or 0)
                ),
                "continuation_toxicity": entry["continuation"]["toxicity"],
            })
    return pd.DataFrame(records)

@st.cache_resource
def load_detoxify():
    return Detoxify("original")

# ── Header ──────────────────────────────────────────────────────────────────

st.title("Toxicity Stress-Test & Moderation Pipeline")
st.caption("RealToxicityPrompts · Llama 3.1 8B · Qwen 2.5 7B · Detoxify scorer")
st.divider()

df = load_results()
baseline_df = load_baseline()

# ── Metric cards ────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Prompts tested", df["prompt_text"].nunique())
col2.metric("Models evaluated", df["model"].nunique())
col3.metric("Responses flagged", int(df["flagged"].sum()))
col4.metric("Avg toxicity (our models)", f"{df['score_toxicity'].mean():.4f}")

st.divider()

# ── Charts ──────────────────────────────────────────────────────────────────

left, right = st.columns(2)

with left:
    st.subheader("Avg Toxicity: Our Models vs Base Model Baseline")

    base_avg = baseline_df["continuation_toxicity"].mean()
    model_avgs = df.groupby("model")["score_toxicity"].mean().reset_index()
    model_avgs.columns = ["label", "avg_toxicity"]
    baseline_row = pd.DataFrame([{"label": "Base Model\n(dataset)", "avg_toxicity": base_avg}])
    chart_df = pd.concat([baseline_row, model_avgs], ignore_index=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c"] + ["#2ecc71"] * len(model_avgs)
    bars = ax.bar(chart_df["label"], chart_df["avg_toxicity"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Avg Toxicity Score")
    ax.set_ylim(0, max(chart_df["avg_toxicity"]) * 1.3)
    ax.set_title("Base Model vs Instruction-Tuned Models")
    for bar, val in zip(bars, chart_df["avg_toxicity"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("Toxicity by Model × Prompt Category")

    pivot = df.pivot_table(
        index="prompt_category", columns="model", values="score_toxicity", aggfunc="mean"
    )
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", linewidths=0.5,
                cbar_kws={"label": "Avg Toxicity"}, ax=ax2)
    ax2.set_title("Toxicity Heatmap")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    fig2.tight_layout()
    st.pyplot(fig2)

st.divider()

# ── Response table ───────────────────────────────────────────────────────────

with st.expander("View all scored responses"):
    st.dataframe(
        df[["prompt_category", "model", "prompt_text", "response_text", "score_toxicity", "flagged"]],
        use_container_width=True,
    )

st.divider()

# ── Live scorer ──────────────────────────────────────────────────────────────

st.subheader("Live Toxicity Scorer")
st.caption("Type any sentence below and score it with Detoxify in real time — no API call, runs locally.")

user_input = st.text_area("Enter text to score:", height=100, placeholder="Type something here...")

if st.button("Score it"):
    if user_input.strip():
        scorer = load_detoxify()
        scores = scorer.predict(user_input)
        score_df = pd.DataFrame([{k: round(float(v), 4) for k, v in scores.items()}])

        toxicity = float(scores["toxicity"])
        if toxicity >= 0.5:
            st.error(f"Flagged — toxicity score: {toxicity:.4f}")
        else:
            st.success(f"Passed — toxicity score: {toxicity:.4f}")

        st.dataframe(score_df, use_container_width=True)
    else:
        st.warning("Please enter some text first.")
