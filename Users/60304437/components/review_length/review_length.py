import os
import pandas as pd
import argparse

def compute_review_length(input_path, output_path):
    # Load reviews (assuming JSON or CSV; adjust as needed)
    df = pd.read_json(os.path.join(input_path, "reviews.json"), lines=True)

    # Compute features
    df["review_length_words"] = df["reviewText"].apply(lambda x: len(str(x).split()))
    df["review_length_chars"] = df["reviewText"].apply(lambda x: len(str(x)))

    # Save output
    os.makedirs(output_path, exist_ok=True)
    df.to_parquet(os.path.join(output_path, "review_length.parquet"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    compute_review_length(args.data, args.out)