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
        pred = list(self.model.forecast(len(x_test)))

        self.df[self.model_name] = fit

        predictions = pd.DataFrame({
            self.model_name: pred,
            self.x_col_name: pd.date_range(self.df[self.x_col_name].max() + pd.Timedelta(days=1),
                                           periods=len(x_test), freq="D")})

        return predictions


if __name__ == "__main__":
    df_alb = pd.read_csv("../data/peajes_alberdi_training_preproc.csv")
    df_alb['date'] = pd.to_datetime(df_alb['fecha'], format='%Y-%m-%d')

    # df_no_nan = df_alb[df_alb['fecha'] < '2019-07-01']
    # df_nan = df_alb[df_alb['fecha'] >= '2019-07-01']
    df_alb = df_alb[df_alb['fecha'] < '2019-07-01']


    model_class = ExponentialSmoothing
    model_instance = model_class(df=df_alb, x_col_name='date', y_col_name='cantidad_pasos', smoothing_level=0.5)
    model_instance.fit_train()
    y_test = model_instance.predict(df_alb[['date']])
    model_instance.plot_results(start_date='2016-01-01', end_date='2016-07-01')