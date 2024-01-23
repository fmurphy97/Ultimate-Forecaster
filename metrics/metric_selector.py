from metrics.errors import ErrorMAE, ErrorMSE, ErrorRMSE, ErrorMAPE, ForecastBiasError, R2Score
from metrics.losses import HuberLoss, LogCoshLoss


def error_selector(error_type):
    error_functions = {
        'MAE': ErrorMAE,
        'MSE': ErrorMSE,
        'RMSE': ErrorRMSE,
        'MAPE': ErrorMAPE,
        'ForecastBias': ForecastBiasError,
        'R2': R2Score,
        "HuberLoss": HuberLoss,
        "LogCosh": LogCoshLoss
    }

    if error_type in error_functions:
        return error_functions[error_type]
    else:
        raise ValueError(f"Invalid error type: {error_type}")
