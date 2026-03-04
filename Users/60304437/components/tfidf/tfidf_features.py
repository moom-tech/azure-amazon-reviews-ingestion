import argparse
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--max_features", type=int, default=1000)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load data
    train_df = pd.read_parquet(args.train_data)
    val_df = pd.read_parquet(args.val_data)
    test_df = pd.read_parquet(args.test_data)
    
    # Initialize TF-IDF
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # IMPORTANT: Fit ONLY on training data
    print("Fitting TF-IDF on training split...")
    train_matrix = tfidf.fit_transform(train_df['reviewText'].fillna(""))
    
    # Transform Val and Test
    val_matrix = tfidf.transform(val_df['reviewText'].fillna(""))
    test_matrix = tfidf.transform(test_df['reviewText'].fillna(""))
    
    # Convert back to DataFrames to merge with existing features (optional but common)
    # Or save as sparse matrices. For this lab, we'll save as parquet files.
    tfidf_cols = [f"tfidf_{i}" for i in range(train_matrix.shape[1])]
    
    def save_split(df, matrix, output_path):
        tfidf_df = pd.DataFrame(matrix.toarray(), columns=tfidf_cols, index=df.index)
        final_df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
        os.makedirs(output_path, exist_ok=True)
        final_df.to_parquet(os.path.join(output_path, "data.parquet"))

    save_split(train_df, train_matrix, args.train_out)
    save_split(val_df, val_matrix, args.val_out)
    save_split(test_df, test_matrix, args.test_out)
    
    print(f"TF-IDF complete. Features created: {len(tfidf_cols)}")

if __name__ == "__main__":
    main()