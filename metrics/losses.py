from metrics.errors import Error
from tensorflow.python.keras.losses import Huber, LogCosh, losses_utils


class Loss(Error):
    def __init__(self, y_true, y_pred, sample_weight, reduction_name="auto"):
        reductions = {"auto": losses_utils.ReductionV2.AUTO,
                      "sum": losses_utils.ReductionV2.SUM,
                      "none": losses_utils.ReductionV2.NONE}

        if reduction_name not in reductions:
            raise ValueError(f"Invalid reduction value: {reduction_name}. Must be one of {['AUTO', 'SUM', 'NONE']}")
        else:
            reduction = reductions[reduction_name]
        super().__init__(y_true, y_pred, sample_weight)
        self.reduction = reduction


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
