from forecasting_models.forecast_model import ForecastModel
from statsmodels.tsa.api import SimpleExpSmoothing

import pandas as pd


class ExponentialSmoothing(ForecastModel):

    def __init__(self, df, x_col_name, y_col_name, smoothing_level):
        """Exponentially Weighted Mean
        Average for last for n days giving more weight to the close days. The weighting coefficient in between 0 and 1

        :param smoothing_level: (float), level average component.
        """
        super().__init__(df, x_col_name, y_col_name)

        self.smoothing_level = smoothing_level
        self.model_name = f"ExpSmooth_{self.smoothing_level}_a"

    def fit_train(self):
        self.model = SimpleExpSmoothing(self.y_train).fit(smoothing_level=self.smoothing_level, optimized=False)

    def predict(self, x_test):
        """Predicts values using the selected dates"""
        # TODO: create method
        fit = list(self.model.fittedvalues)
        values_to_forecast = len(x_test)
        pred = list(self.model.forecast(values_to_forecast))

        full_pred = fit + pred

        # Calculate the start date for the new date range by adding one day to the maximum date
        start_date = self.df[self.x_col_name].max() + pd.Timedelta(days=1)
        # Generate the date range
        new_date_range = pd.date_range(start=start_date, periods=values_to_forecast, freq="D")
        # Create a new DataFrame with the date range
        date_range_df = pd.DataFrame({self.x_col_name: new_date_range})

        # Concatenate the original DataFrame with the new date range DataFrame
        result_df = pd.concat([self.df[[self.x_col_name]], date_range_df], ignore_index=True)

        result_df[self.model_name] = full_pred

        self.df = pd.merge(self.df, result_df, on=self.x_col_name, how="outer")

        return result_df


if __name__ == "__main__":
    df_alb = pd.read_csv("../data/peajes_alberdi_training_preproc.csv")
    df_alb['date'] = pd.to_datetime(df_alb['fecha'], format='%Y-%m-%d')

    df_alb = df_alb[df_alb['fecha'] < '2019-07-01']


    model_class = ExponentialSmoothing
    model_instance = model_class(df=df_alb, x_col_name='date', y_col_name='cantidad_pasos', smoothing_level=0.5)
    model_instance.fit_train()
    y_test = model_instance.predict(df_alb[['date']])
    model_instance.plot_results(start_date='2019-01-01', end_date='2020-07-01')