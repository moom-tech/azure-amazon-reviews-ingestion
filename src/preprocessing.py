import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

COLUMNS = (
    ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] +
    [f'sensor_{i}' for i in range(1, 22)]
)

def load_data(data_dir):
    filepath = os.path.join(data_dir, 'train_FD001.txt')
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMNS)
    print(f"Loaded: {df.shape} | Engines: {df['engine_id'].nunique()}")
    return df

def add_rul(df):
    """Compute Remaining Useful Life for each row."""
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)
    return df

def drop_constant_sensors(df):
    """Remove sensors with zero variance — they carry no information."""
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    available = [c for c in sensor_cols if c in df.columns]
    std = df[available].std()
    constant = std[std == 0].index.tolist()
    print(f"Dropping constant sensors: {constant}")
    return df.drop(columns=constant)

def normalize_sensors(df):
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df, scaler

def preprocess_pipeline(data_dir, debug=False):
    df = load_data(data_dir)
    if debug:
        # Use only 20 engines for fast local testing
        engine_subset = df['engine_id'].unique()[:20]
        df = df[df['engine_id'].isin(engine_subset)]
        print(f"DEBUG MODE: using {df['engine_id'].nunique()} engines")
    df = add_rul(df)
    df = drop_constant_sensors(df)
    df, scaler = normalize_sensors(df)
    print(f"Preprocessing complete. Shape: {df.shape}")
    return df, scaler