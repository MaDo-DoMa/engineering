import numpy as np

# Default hyperparameters (used only when tuning is skipped, e.g. tiny
# training segments) - same values the training pipeline used originally.
DEFAULT_PROCESS_VAR = 1e-4
DEFAULT_MEASUREMENT_VAR = 1e-2

# Candidate process variances for the Q grid search (log-spaced). Note:
# when the optimum sits at the smallest grid value the data prefers a
# nearly static level, i.e. the filter behaves like the Mean Return
# baseline - an honest empirical finding, not a bug.
Q_GRID = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


class KalmanFilterModel:

    def __init__(self, process_var=DEFAULT_PROCESS_VAR, measurement_var=DEFAULT_MEASUREMENT_VAR):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.post_mean = 0.0
        self.post_var = 1.0

    def fit(self, y_train):
        self.post_mean = np.mean(y_train)
        self.post_var = 1.0

        for z in y_train:
            self._update_state(z)

        return self

    def _update_state(self, z):
        prior_mean = self.post_mean
        prior_var = self.post_var + self.process_var

        kalman_gain = prior_var / (prior_var + self.measurement_var)

        self.post_mean = prior_mean + kalman_gain * (z - prior_mean)
        self.post_var = (1 - kalman_gain) * prior_var

    def predict(self, y_test):
        preds = []
        for z in y_test:
            preds.append(self.post_mean)

            self._update_state(z)

        return np.array(preds)


def tune_kalman_params(train_returns, q_grid=Q_GRID, val_frac=0.2):
    """
    Estimates the Kalman filter hyperparameters from TRAINING data only -
    nothing outside the training segment is ever touched.

      - R (measurement variance) = variance of the observed training log
        returns, i.e. the empirical noise level of the return series.
      - Q (process variance) = grid search on the FINAL `val_frac` of the
        training segment, mirroring the RF/NN hyperparameter protocol
        (validate on the last 20% of train). For every candidate Q the
        filter is fit on the first (1 - val_frac) of the training returns
        and scored by one-step-ahead MSE on the held-out final segment;
        each prediction is emitted BEFORE the filter sees its observation,
        so the tuning itself is free of lookahead.

    Returns (process_var, measurement_var).
    """
    y = np.asarray(train_returns, dtype=float)
    r = float(np.var(y))
    if not np.isfinite(r) or r <= 0:
        r = DEFAULT_MEASUREMENT_VAR

    n_fit = max(1, int(len(y) * (1.0 - val_frac)))
    y_fit, y_val = y[:n_fit], y[n_fit:]
    if len(y_val) < 10:
        return DEFAULT_PROCESS_VAR, r

    best_q, best_mse = DEFAULT_PROCESS_VAR, float("inf")
    for q in q_grid:
        f = KalmanFilterModel(process_var=q, measurement_var=r).fit(y_fit)
        preds = f.predict(y_val)
        mse = float(np.mean((preds - y_val) ** 2))
        if mse < best_mse:
            best_mse, best_q = mse, q
    return best_q, r