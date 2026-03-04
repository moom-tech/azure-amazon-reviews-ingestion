import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    
    # Load pre-trained transformer model
    # 'all-MiniLM-L6-v2' is small, fast, and high-quality
    model = SentenceTransformer(args.model_name)
    
    print(f"Generating embeddings using {args.model_name}...")
    # Convert reviews to a list and encode
    reviews = df['reviewText'].fillna("").tolist()
    embeddings = model.encode(reviews, show_progress_bar=True)
    
    # Convert embeddings to a DataFrame
    embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
    
    # Combine with original data
    final_df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    
    os.makedirs(args.output, exist_ok=True)
    final_df.to_parquet(os.path.join(args.output, "data.parquet"))
    print(f"Embeddings complete. Dimension: {embeddings.shape[1]}")

if __name__ == "__main__":
    main()