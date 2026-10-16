import joblib
from sklearn.ensemble import RandomForestRegressor
from config import ARTIFACTS_DIR, RANDOM_SEED

def train_and_save_rf(X_train, y_train):
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print("Training RF model...")
    rf_model.fit(X_train, y_train)

    model_path = ARTIFACTS_DIR / "rf_model.pkl"
    joblib.dump(rf_model, model_path)
    print("Model trained and saved at " + str(model_path))

    return rf_model