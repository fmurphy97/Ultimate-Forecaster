import pandas as pd
from forecasting_models.forecast_model import ForecastModel
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


class TripleExponentialSmoothing(ForecastModel):

    def __init__(self, df, x_col_name, y_col_name,
                 smoothing_level=0.0, smoothing_trend=0.0, smoothing_seasonal=0.0,
                 trend="additive", seasonal="additive", seasonal_periods=364):
        """Triple Exponential Smoothing Forecast Model, also known as Holt-Winters

        This model extends the ForecastModel class and implements triple exponential smoothing using the
        statsmodels library.

        Parameters:
            smoothing_level (float): The smoothing parameter for the level component.
            smoothing_trend (float): The smoothing parameter for the trend component.
            smoothing_seasonal (float): The smoothing parameter for the seasonal component.
            trend (str, optional): The type of trend component, either 'additive' or 'multiplicative'.
                Defaults to 'additive'.
            seasonal (str, optional): The type of seasonal component, either 'additive' or 'multiplicative'.
                Defaults to 'additive'.
            seasonal_periods (int, optional): The number of periods in a complete seasonal cycle. Defaults to 364.
        """
        super().__init__(df, x_col_name, y_col_name)

        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal

        self.model_name = (f"ExpSmooth"
                           f"_{int(100 * self.smoothing_level)}"
                           f"_{int(100 * self.smoothing_trend)}"
                           f"_{int(100 * self.smoothing_seasonal)}")

        self.trend_type = trend
        self.seasonal_type = seasonal
        self.seasonal_periods = seasonal_periods

    def fit_train(self):
        self.model = ExponentialSmoothing(self.y_train, seasonal_periods=self.seasonal_periods, trend=self.trend_type,
                                          seasonal=self.seasonal_type). \
            fit(smoothing_level=self.smoothing_level,
                smoothing_trend=self.smoothing_trend,
                smoothing_seasonal=self.smoothing_seasonal,
                optimized=False)

    def predict(self, x_test):
        # The fitted values will be the ones used for the train data
        fit = list(self.model.fittedvalues)

        # Forecast n periods using the x_test df
        values_to_forecast = len(x_test)
        pred = list(self.model.forecast(values_to_forecast))
        pred = [x if x >= 0 else 0 for x in pred]

        # Place the forecasted and fitted values in a new df
        date_range_df = x_test[[self.x_col_name]]
        result_df = pd.concat([self.df[[self.x_col_name]], date_range_df], ignore_index=True)
        result_df[self.model_name] = fit + pred

        # Merge train df with predicted df
        self.df = pd.merge(self.df, result_df, on=self.x_col_name, how="outer")

        return result_df[-values_to_forecast:]


class DoubleExponentialSmoothing(TripleExponentialSmoothing):
    def __init__(self, df, x_col_name, y_col_name, smoothing_level, smoothing_trend):
        """Double Exponential Smoothing Forecast Model also known as Holt
        This model extends the TripleExponentialSmoothing class and implements double exponential smoothing using the
        statsmodels library.
        """

        super().__init__(df=df, x_col_name=x_col_name, y_col_name=y_col_name, smoothing_level=smoothing_level,
                         smoothing_trend=smoothing_trend)

    def fit_train(self):
        self.model = Holt(self.y_train). \
            fit(smoothing_level=self.smoothing_level,
                smoothing_trend=self.smoothing_trend,
                optimized=False)


class SimpleExponentialSmoothing(TripleExponentialSmoothing):
    """Simple Exponential Smoothing Forecast Model.
    This model extends the TripleExponentialSmoothing class and implements simple exponential smoothing using the
    statsmodels library.
    """

    def __init__(self, df, x_col_name, y_col_name, smoothing_level):
        super().__init__(df=df, x_col_name=x_col_name, y_col_name=y_col_name, smoothing_level=smoothing_level)

    def fit_train(self):
        self.model = SimpleExpSmoothing(self.y_train). \
            fit(smoothing_level=self.smoothing_level,
                optimized=False)


class OptimizedExponentialSmoothing(TripleExponentialSmoothing):
    def __init__(self, df, x_col_name, y_col_name, trend="additive", seasonal="additive", seasonal_periods=364):
        super().__init__(df=df, x_col_name=x_col_name, y_col_name=y_col_name, trend=trend, seasonal=seasonal,
                         seasonal_periods=seasonal_periods)

        self.model_name = "ExpSmooth_opt"

    def update_model_name(self):
        self.model_name = (
            f"ExpSmoothOpt"
            f"_{int(100 * self.model.params['smoothing_level'])}"
            f"_{int(100 * self.model.params['smoothing_trend'])}"
            f"_{int(100 * self.model.params['smoothing_seasonal'])}")

    def fit_train(self):
        self.model = ExponentialSmoothing(self.y_train, seasonal_periods=self.seasonal_periods, trend=self.trend_type,
                                          seasonal=self.seasonal_type). \
            fit(optimized=True)

        self.update_model_name()
