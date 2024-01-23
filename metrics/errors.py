from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error, r2_score


class Error:
    def __init__(self, y_true, y_pred, sample_weight=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sample_weight = sample_weight

    def calculate_error(self):
        pass


class ErrorMAE(Error):
    """Mean absolute error"""

    def calculate_error(self):
        return mean_absolute_error(y_true=self.y_true, y_pred=self.y_pred, sample_weight=self.sample_weight)


class ErrorMSE(Error):
    """Mean square error"""

    def calculate_error(self):
        return mean_squared_error(y_true=self.y_true, y_pred=self.y_pred, sample_weight=self.sample_weight)


class ErrorRMSE(Error):
    """Root mean squared error"""

    def calculate_error(self):
        return root_mean_squared_error(y_true=self.y_true, y_pred=self.y_pred, sample_weight=self.sample_weight)


class ErrorMAPE(Error):
    """Mean absolute percent error"""

    def calculate_error(self):
        return mean_absolute_percentage_error(y_true=self.y_true, y_pred=self.y_pred, sample_weight=self.sample_weight)


class ForecastBiasError(Error):
    """Forecast bias"""

    def calculate_error(self):
        if self.sample_weight is not None:
            return (self.sample_weight * (self.y_pred - self.y_true)).sum() / self.sample_weight.sum()
        else:
            return (self.y_pred - self.y_true).mean()


class R2Score(Error):
    """Coefficient of determination, regression score function."""

    def calculate_error(self):
        return r2_score(y_true=self.y_true, y_pred=self.y_pred, sample_weight=self.sample_weight)
