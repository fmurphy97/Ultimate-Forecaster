from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error
from tensorflow.keras.losses import Huber, LogCosh, losses_utils, Reduction


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
            return (self.weights * (self.ypred - self.ytrue)).sum() / self.weights.sum()
        else:
            return (self.ypred - self.ytrue).mean()


class Loss(Error):
    def __init__(self, y_true, y_pred, sample_weight, reduction=losses_utils.ReductionV2.AUTO):
        if reduction not in [losses_utils.ReductionV2.AUTO, Reduction.SUM, Reduction.NONE]:
            raise ValueError(f"Invalid reduction value: {reduction}. Must be one of {['AUTO', 'SUM', 'NONE']}")
        super().__init__(y_true, y_pred, sample_weight)
        self.reduction = reduction


class HuberLoss(Loss):
    """Huber Loss"""

    def calculate_error(self):
        loss = Huber()
        if self.sample_weight is not None:
            return loss(y_true=self.y_true, y_pred=self.y_pred, sample_weight=self.sample_weight, reduction=self.reduction)
        else:
            return loss(y_true=self.y_true, y_pred=self.y_pred, reduction=self.reduction)


class LogCoshLoss(Loss):
    """Computes the logarithm of the hyperbolic cosine of the prediction error."""

    def calculate_error(self):
        loss = LogCosh()
        if self.sample_weight is not None:
            return loss(y_true=self.y_true, y_pred=self.y_pred, sample_weight=self.sample_weight, reduction=self.reduction)
        else:
            return loss(y_true=self.y_true, y_pred=self.y_pred, reduction=self.reduction)


def error_selector(error_type, y_true, y_pred, sample_weight=None, reduction=losses_utils.ReductionV2.AUTO):
    if error_type == 'MAE':
        return ErrorMAE(y_true, y_pred, sample_weight)
    elif error_type == 'MSE':
        return ErrorMSE(y_true, y_pred, sample_weight)
    elif error_type == 'RMSE':
        return ErrorRMSE(y_true, y_pred, sample_weight)
    elif error_type == 'MAPE':
        return ErrorMAPE(y_true, y_pred, sample_weight)
    elif error_type == 'ForecastBias':
        return ForecastBiasError(y_true, y_pred, sample_weight)
    elif error_type == 'HuberLoss':
        return HuberLoss(y_true, y_pred, sample_weight, reduction)
    elif error_type == 'LogCoshLoss':
        return LogCoshLoss(y_true, y_pred, sample_weight, reduction)
    else:
        raise ValueError(f"Invalid error type: {error_type}")
