import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression

def remove_low_variance(X, threshold=0.001):  # much lower than before
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    kept = X.columns[selector.get_support()]
    print(f"Variance filter:     {X.shape[1]:>5} → {len(kept)} features")
    return pd.DataFrame(X_reduced, columns=kept, index=X.index)

def remove_high_correlation(X, threshold=0.98):  # raised from 0.85 to 0.98
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    print(f"Correlation filter:  {X.shape[1]:>5} → {X_reduced.shape[1]} features")
    return X_reduced

def mutual_info_filter(X, y, top_k=50):
    # If fewer features than top_k, just return all of them
    if X.shape[1] <= top_k:
        print(f"Mutual info filter:  {X.shape[1]:>5} → {X.shape[1]} features (all kept)")
        return X, pd.Series(np.ones(X.shape[1]), index=X.columns)
    mi = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns)
    top = mi_series.nlargest(top_k).index.tolist()
    print(f"Mutual info filter:  {X.shape[1]:>5} → {len(top)} features")
    return X[top], mi_series

def apply_filter_pipeline(X, y, variance_thresh=0.001, corr_thresh=0.98, top_k=50):
    print("\n--- Filter-Based Selection ---")
    X = remove_low_variance(X, variance_thresh)
    X = remove_high_correlation(X, corr_thresh)
    X, mi_scores = mutual_info_filter(X, y, top_k)
    return X, mi_scores