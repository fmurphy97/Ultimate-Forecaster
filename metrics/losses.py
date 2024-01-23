from metrics.errors import Error
from tensorflow.python.keras.losses import Huber, LogCosh


class Loss(Error):
    def __init__(self, y_true, y_pred, sample_weight):
        super().__init__(y_true, y_pred, sample_weight)


class HuberLoss(Loss):
    """Huber Loss"""

    def calculate_error(self):
        loss = Huber()
        if self.sample_weight is not None:
            return loss(y_true=self.y_true, y_pred=self.y_pred, sample_weight=self.sample_weight)
        else:
            return loss(y_true=self.y_true, y_pred=self.y_pred)


class LogCoshLoss(Loss):
    """Computes the logarithm of the hyperbolic cosine of the prediction error."""

    def calculate_error(self):
        loss = LogCosh()
        if self.sample_weight is not None:
            return loss(y_true=self.y_true, y_pred=self.y_pred, sample_weight=self.sample_weight)
        else:
            return loss(y_true=self.y_true, y_pred=self.y_pred)
