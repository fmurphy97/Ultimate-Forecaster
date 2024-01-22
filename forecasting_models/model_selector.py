from forecasting_models.naive_model import NaiveForecastModel
from forecasting_models.rolling_avg_model import RollingMean
from forecasting_models.exponential_smoothing import (SimpleExponentialSmoothing, DoubleExponentialSmoothing,
                                                      TripleExponentialSmoothing, OptimizedExponentialSmoothing)


def select_forecasting_model(model_name):
    model_name = model_name.lower()

    model_classes = {
        'naive': NaiveForecastModel,
        'rolling_mean': RollingMean,
        'exponential_smoothing': SimpleExponentialSmoothing,
        'holt': DoubleExponentialSmoothing,
        'holt_winters': TripleExponentialSmoothing,
        'optimized_exponential_smoothing': OptimizedExponentialSmoothing,
        # 'autoregressive': AutoRegressiveModel,
        # 'arima': ARIMAModel,
        # 'sarima': SARIMAModel,
        # 'sarimax': SARIMAXModel,
        # 'multiple linear regression': MultipleLinearRegressionModel,
        # 'lightgbm': LightGBM,
        # 'prophet': ProphetModel,
        # 'lstm': LSTMModel,
        # 'xgboost': XGBoostModel
    }

    if model_name in model_classes:
        return model_classes[model_name]
    else:
        raise ValueError("Invalid model name")
