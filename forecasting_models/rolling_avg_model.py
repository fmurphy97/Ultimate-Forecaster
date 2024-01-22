from forecasting_models.forecast_model import ForecastModel
import pandas as pd


class RollingMean(ForecastModel):

    def __init__(self, df, x_col_name, y_col_name, n_periods):
        """Rolling mean, also known as moving averages (MA).
        Takes "n" previous target variable and averages them and returns as a new value.
        Commonly used periods are: 3 (3 day rolling mean), 7 (weekly rolling average), and 364 (yearly rolling average)

        :param n_periods: (int)
            The number of previous periods to be used for calculating the rolling mean.
        """
        super().__init__(df, x_col_name, y_col_name)
        self.n_periods = n_periods

        self.model_name = f"MovingAverage_{self.n_periods}_periods"

    def predict(self, x_test):
        self.df = pd.concat([self.df, x_test[[self.x_col_name]]], axis=0, ignore_index=True)

        # Create the rolling average
        self.df[self.model_name] = self.df[self.y_col_name].transform(
            lambda x: x.rolling(self.n_periods, 1).mean())

        # Fill missing values
        self.df[self.model_name] = self.df[self.model_name].shift(1).bfill()
        self.df[self.model_name] = self.df[self.model_name].shift(1).ffill()

        values_to_forecast = len(x_test)
        return self.df[[self.x_col_name, self.model_name]][-values_to_forecast:]
