from forecast_model import ForecastModel
import pandas as pd


class RollingMean(ForecastModel):
    """Rolling mean, also known as moving averages (MA).
    Takes "n" previous target variable and averages them and returns as a new value.
    Commonly used periods are: 3 (3 day rolling mean), 7 (weekly rolling average), and 364 (yearly rolling average)
    """

    def __init__(self, df, x_col_name, y_col_name, n_periods):
        super().__init__(df, x_col_name, y_col_name)
        self.n_periods = n_periods

        self.model_name = f"MovingAverage_{self.n_periods}_periods"

    def fit_train(self):
        # Create the rolling average
        self.df[self.model_name] = self.y_train.transform(
            lambda x: x.rolling(self.n_periods, 1).mean())

        # Fill missing values
        self.df[self.model_name] = self.df[self.model_name].shift(1).ffill()
        self.df[self.model_name] = self.df[self.model_name].shift(1).bfill()

    def predict(self, x_test):
        # Return only the required rows
        return pd.merge(self.df[[self.x_col_name, self.model_name]],
                        x_test, how="right", on=self.x_col_name)
