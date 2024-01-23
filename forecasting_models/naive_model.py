from forecasting_models.forecast_model import ForecastModel
import pandas as pd


class NaiveForecastModel(ForecastModel):

    def __init__(self, df, x_col_name, y_col_name, n_periods):
        """
        Lagged feature also known as Naïve model.
        Copies the y value but shifted lag_periods days.
        Commonly used lags are: 1 (Naïve Last Value), 7 (Naïve Last Week) and 364 (Naïve Last Cycle)

        :param n_periods: int
            The lag period indicating how many periods (in general days) to shift the target variable for forecasting.
        """
        super().__init__(df, x_col_name, y_col_name)
        self.n_periods = n_periods

        self.model_name = f"Naive_lag_{self.n_periods}"

    def predict(self, x_test):
        self.df = pd.concat([self.df, x_test[[self.x_col_name]]], axis=0, ignore_index=True)

        self.df[self.model_name] = self.df[self.y_col_name].shift(self.n_periods)
        self.df[self.model_name] = self.df[self.model_name].ffill()

        values_to_forecast = len(x_test)

        fit = self.df[self.model_name][:-values_to_forecast]
        predict = self.df[self.model_name][-values_to_forecast:]
        return list(fit), list(predict)
