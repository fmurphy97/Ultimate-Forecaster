from forecasting_models.forecast_model import ForecastModel
import pandas as pd


class NaiveForecastModel(ForecastModel):

    def __init__(self, df, x_col_name, y_col_name, n_periods):
        """Lagged feature.
        Copies the y value but shifted lag_periods days.
        Commonly used lags are: 1 (Naïve Last Value), 7 (Naïve Last Week) and 364 (Naïve Last Cycle)

        :param n_periods:
        """
        super().__init__(df, x_col_name, y_col_name)
        self.n_periods = n_periods

        self.model_name = f"Naive_lag_{self.n_periods}"

    def fit_train(self):
        self.df[self.model_name] = self.y_train.shift(self.n_periods)

    def predict(self, x_test):
        """Predicts values using the selected dates"""
        return pd.merge(self.df[[self.x_col_name, self.model_name]],
                        x_test, how="right", on=self.x_col_name)
