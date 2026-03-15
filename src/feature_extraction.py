import time
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

def prepare_tsfresh_input(df):
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    # Explicitly exclude RUL, op_settings — only sensors + id + time
    ts_data = df[['engine_id', 'cycle'] + sensor_cols].copy()
    print(f"tsfresh input shape: {ts_data.shape}")  # Should be (20631, ~20)
    return ts_data

def extract_tsfresh_features(df, settings='efficient', n_jobs=1):
    """
    settings='minimal'   → fast, fewer features  (good for testing)
    settings='efficient' → slower, more features  (better accuracy)
    n_jobs=1             → single-threaded (use -1 for all cores on Azure)
    """
    ts_data = prepare_tsfresh_input(df)
    fc_params = (MinimalFCParameters() if settings == 'minimal'
                 else EfficientFCParameters())

    print(f"Starting tsfresh extraction | settings={settings} | n_jobs={n_jobs}")
    start = time.time()

    features = extract_features(
        ts_data,
        column_id='engine_id',
        column_sort='cycle',
        default_fc_parameters=fc_params,
        n_jobs=n_jobs,
        impute_function=impute,
        show_warnings=False,
        disable_progressbar=False
    )

    elapsed = time.time() - start
    print(f"Extraction done: {features.shape} in {elapsed:.2f}s")
    return features, elapsed

def get_rul_labels(df):
    """One RUL label per engine — max RUL (total useful life at start)."""
    return df.groupby('engine_id')['RUL'].max()  # Total useful life at start