import optuna
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import ARTIFACTS_DIR, RANDOM_SEED

def objective(trial, X_train, y_train, X_val, y_val):

    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    #Minimalize MSE in validation set
    return mean_squared_error(y_val, prediction)

def tune_and_save_rf(X_train, y_train, X_val, y_val, n_trials=20):
    print(f"tuning random forest")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    print(f"\nOptimization finished")
    print(f"Best parameters: {study.best_params}")
    print(f"Best validation MSE: {study.best_value}")

    #Training final model with the best parameters
    print(f"Training final model of random forest")
    best_rf = RandomForestRegressor(
        **study.best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    best_rf.fit(X_train, y_train)

    model_path=ARTIFACTS_DIR / "rf_model_optimized.pkl"
    joblib.dump(best_rf, model_path)
    print(f"Optimized model saved at "+ str(model_path))

    return best_rf