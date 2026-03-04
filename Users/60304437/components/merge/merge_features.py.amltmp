import argparse
import os
import pandas as pd
from functools import reduce

def parse_args():
    parser = argparse.ArgumentParser()
    # Define inputs for all feature sources
    parser.add_argument("--length_data", type=str, required=True)
    parser.add_argument("--sentiment_data", type=str, required=True)
    parser.add_argument("--tfidf_data", type=str, required=True)
    parser.add_argument("--embedding_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load all datasets
    # We assume 'asin' and 'reviewerID' are the join keys
    df_len = pd.read_parquet(args.length_data)
    df_sent = pd.read_parquet(args.sentiment_data)
    df_tfidf = pd.read_parquet(args.tfidf_data)
    df_emb = pd.read_parquet(args.embedding_data)
    
    # List of dataframes to merge
    dfs = [df_len, df_sent, df_tfidf, df_emb]
    
    # Use reduce to perform a sequential left join on the keys
    # This prevents column duplication for the keys themselves
    print("Merging all features...")
    df_final = reduce(lambda left, right: pd.merge(
        left, right, on=['asin', 'reviewerID'], how='left', suffixes=('', '_drop')
    ), dfs)

    # Remove any columns duplicated during the merge process (suffix '_drop')
    df_final = df_final.loc[:, ~df_final.columns.str.contains('_drop')]
    
    os.makedirs(args.output, exist_ok=True)
    df_final.to_parquet(os.path.join(args.output, "data.parquet"))
    print(f"Merge complete! Final feature count: {len(df_final.columns)}")
    print(f"Total rows: {len(df_final)}")

if __name__ == "__main__":
    main()