import pandas as pd
from moderator import moderate_dataframe

INPUT_FILE = "scored_outputs.csv"
OUTPUT_FILE = "moderated_outputs.csv"
THRESHOLD = 0.5
STRATEGY = "block"  # change to "rewrite" to have Ollama regenerate flagged responses

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    print(f"Strategy: {STRATEGY} | Threshold: {THRESHOLD}\n")

    df = moderate_dataframe(df, strategy=STRATEGY, threshold=THRESHOLD)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

    print("\n--- Moderation summary ---")
    print(f"Total responses:  {len(df)}")
    print(f"Flagged:          {df['flagged'].sum()}")
    print(f"Passed:           {(~df['flagged']).sum()}")

if __name__ == "__main__":
    main()
