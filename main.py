import argparse
import os
import time
import json
import glob
import pandas as pd
from azureml.core.run import Run

from src.preprocessing      import preprocess_pipeline
from src.feature_extraction import extract_tsfresh_features, get_rul_labels
from src.feature_selection  import apply_filter_pipeline
from src.genetic_algorithm  import apply_ga_selection
from src.model              import train_and_evaluate, MODELS
from src.evaluate           import compute_metrics, plot_predictions

# ── Parse arguments (Azure passes these in) ─────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',   default='data/raw',  help='Dataset directory')
parser.add_argument('--output_dir', default='results',   help='Output directory')
parser.add_argument('--settings',   default='efficient', help='tsfresh: minimal/efficient')
parser.add_argument('--top_k',      default=100, type=int)
parser.add_argument('--ga_gen',     default=20,  type=int)
parser.add_argument('--ga_pop',     default=50,  type=int)
parser.add_argument('--debug',  action='store_true', help='Use subset for fast testing')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
print(f"Data dir:   {args.data_dir}")
print(f"Output dir: {args.output_dir}")

# Verify data files are actually there at runtime
found = glob.glob(os.path.join(args.data_dir, '*.txt'))
print(f"Data files found: {len(found)} file(s)")
if not found:
    raise FileNotFoundError(f"No .txt files found in {args.data_dir}. "
                            f"Check dataset registration and path.")

# Azure ML run context — logs metrics to Studio automatically
run = Run.get_context()
total_start = time.time()

# ── 1. Preprocess ────────────────────────────────────────────────────────────
print("\n========== PREPROCESSING ==========")
df, scaler = preprocess_pipeline(args.data_dir, debug=args.debug)

# ── 2. Feature Extraction ────────────────────────────────────────────────────
print("\n========== FEATURE EXTRACTION ==========")
features, extraction_time = extract_tsfresh_features(df, settings=args.settings)
y = get_rul_labels(df)
features, y = features.align(y, join='inner', axis=0)
run.log('extraction_time_s', round(extraction_time, 2))
run.log('features_after_extraction', features.shape[1])

# ── 3. Filter Selection ──────────────────────────────────────────────────────
print("\n========== FILTER SELECTION ==========")
X_filtered, mi_scores = apply_filter_pipeline(features, y, top_k=args.top_k)
run.log('features_after_filter', X_filtered.shape[1])

# ── 4. Genetic Algorithm ─────────────────────────────────────────────────────
print("\n========== GENETIC ALGORITHM ==========")
X_final, selected_features, ga_log = apply_ga_selection(X_filtered, y)
run.log('features_after_ga', X_final.shape[1])

# Save selected features for future labs
pd.Series(selected_features).to_csv(
    os.path.join(args.output_dir, 'selected_features.csv'), index=False)
X_final.to_csv(os.path.join(args.output_dir, 'final_feature_matrix.csv'))

# ── 5. Train and Evaluate All Models ─────────────────────────────────────────
print("\n========== MODEL TRAINING ==========")
all_results = {}
for model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
    model, y_pred, y_test, train_time = train_and_evaluate(X_final, y, model_name)
    metrics = compute_metrics(y_test, y_pred)
    metrics['train_time_s']  = round(train_time, 2)
    metrics['n_features']    = X_final.shape[1]
    all_results[model_name]  = metrics

    # Log each model's metrics to Azure ML Studio
    run.log(f'{model_name}_RMSE', metrics['RMSE'])
    run.log(f'{model_name}_R2',   metrics['R2'])
    run.log(f'{model_name}_MAE',  metrics['MAE'])

    plot_predictions(
        y_test, y_pred, model_name,
        os.path.join(args.output_dir, f'pred_{model_name}.png')
    )
    print(f"{model_name:>18}: RMSE={metrics['RMSE']:.2f} | "
          f"R2={metrics['R2']:.3f} | Features={metrics['n_features']}")

# ── 6. Save and Log Summary ───────────────────────────────────────────────────
total_time = time.time() - total_start
run.log('total_pipeline_time_s', round(total_time, 2))

results_df = pd.DataFrame(all_results).T
results_df.to_csv(os.path.join(args.output_dir, 'results_summary.csv'))

with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)

print(f"\n========== DONE ==========")
print(f"Total time:          {total_time:.2f}s")
print(f"Extraction time:     {extraction_time:.2f}s")
print(f"Final feature count: {X_final.shape[1]}")
print(results_df.to_string())