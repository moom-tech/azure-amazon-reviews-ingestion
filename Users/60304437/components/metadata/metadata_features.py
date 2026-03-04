import argparse
import os
import pandas as pd
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    
    # Ensure reviewText is string
    text_col = df['reviewText'].fillna("").astype(str)
    
    # 1. Word Count
    df['word_count'] = text_col.apply(lambda x: len(x.split()))
    
    # 2. Character Count (Length)
    df['char_count'] = text_col.apply(len)
    
    # 3. Average Word Length
    df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)
    
    # 4. Exclamation/Question Mark Count (Signal of strong opinion)
    df['punc_intensity'] = text_col.apply(lambda x: len(re.findall(r'[!?]', x)))
    
    # 5. ALL CAPS ratio (Signal of "shouting")
    df['caps_ratio'] = text_col.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))

    os.makedirs(args.output, exist_ok=True)
    df.to_parquet(os.path.join(args.output, "data.parquet"))
    print(f"Metadata features generated for {len(df)} rows.")

if __name__ == "__main__":
    main()