from forecasting_models.naive_model import NaiveForecastModel
from forecasting_models.rolling_avg_model import RollingMean
from forecasting_models.exponential_smoothing import ExponentialSmoothing


def select_forecasting_model(model_name):
    model_name = model_name.lower()

    model_classes = {
        'naive': NaiveForecastModel,
        'rolling mean': RollingMean,
        'exponential smoothing': ExponentialSmoothing,
        # 'Holt': DoubleExponentialSmoothingModel,
        # 'Holt/Winters': TripleExponentialSmoothingModel,
        # 'optimized exponential smoothing': OptimizedExponentialSmoothingModel,
        # 'autoregressive': AutoRegressiveModel,
        # 'arima': ARIMAModel,
        # 'sarima': SARIMAModel,
        # 'sarimax': SARIMAXModel,
        # 'multiple linear regression': MultipleLinearRegressionModel,
        # 'lightgbm': LightGBMModel,
        # 'prophet': ProphetModel,
        # 'lstm': LSTMModel,
        # 'xgboost': XGBoostModel
    }

    if model_name in model_classes:
        return model_classes[model_name]
    else:
        raise ValueError("Invalid model name")
