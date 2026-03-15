## Azure Run Details
- **Run name:** serene_celery_v53ggvhn
- **Date:** March 15, 2026
- **Duration:** 5m 55s
- **Compute:** Standard_E4ds_v4
- **Settings:** tsfresh EfficientFCParameters

## Pipeline Summary
| Stage | Input Features | Output Features |
|-------|---------------|----------------|
| tsfresh Extraction | 19 sensors | 750+ |
| Variance Filter (0.001) | 750+ | ~600 |
| Correlation Filter (0.98) | ~600 | ~200 |
| Mutual Info (top 50) | ~200 | 50 |
| Genetic Algorithm | 50 | 12 |

## Results
| Model | RMSE | MAE | R2 | Features | Train Time |
|-------|------|-----|----|---------|------------|
| RandomForest | 7.6469 | 4.916 | 0.9661 | 12 | 0.35s |
| XGBoost | 7.834 | 4.5869 | 0.9644 | 12 | 2.20s |
| LightGBM | 9.0173 | 5.5275 | 0.9529 | 12 | 0.26s |

**Best model: RandomForest** (RMSE=7.65, R2=0.9661)