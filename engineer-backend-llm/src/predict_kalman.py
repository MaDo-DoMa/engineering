import numpy as np

class KalmanFilterModel:

    def __init__(self, process_var=1e-4, measurement_var=1e-2):
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