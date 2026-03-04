import argparse
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    
    analyzer = SentimentIntensityAnalyzer()
    
    # Apply VADER sentiment analysis
    # We use .get() to handle potential empty strings safely
    sentiments = df['reviewText'].fillna("").apply(lambda x: analyzer.polarity_scores(x))
    
    df['sentiment_pos'] = sentiments.apply(lambda x: x['pos'])
    df['sentiment_neg'] = sentiments.apply(lambda x: x['neg'])
    df['sentiment_neu'] = sentiments.apply(lambda x: x['neu'])
    df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
    
    os.makedirs(args.output, exist_ok=True)
    df.to_parquet(os.path.join(args.output, "data.parquet"))
    print(f"Processed {len(df)} rows with sentiment scores.")

if __name__ == "__main__":
    main()