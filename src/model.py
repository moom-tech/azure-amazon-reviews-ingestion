import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

MODELS = {
    'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost':          XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'LightGBM':         LGBMRegressor(
        n_estimators=100, random_state=42, n_jobs=-1,
        min_child_samples=1  # allows training on small datasets
    ),
}

def train_and_evaluate(X, y, model_name='XGBoost'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = MODELS[model_name]
    t0 = time.time()
    model.fit(X_train, y_train)
    return model, model.predict(X_test), y_test, time.time() - t0