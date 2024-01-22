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


if __name__ == "__main__":
    df_alb = pd.read_csv("../data/peajes_alberdi_training_preproc.csv")
    df_alb['date'] = pd.to_datetime(df_alb['fecha'], format='%Y-%m-%d')
    df_alb = df_alb[["date", "cantidad_pasos"]]

    df_alb_train = df_alb[df_alb['date'] < '2019-07-01']
    df_alb_test = df_alb[df_alb['date'] >= '2019-07-01']

    model_class = RollingMean
    model_instance = model_class(df=df_alb_train, x_col_name='date', y_col_name='cantidad_pasos', n_periods=7)
    model_instance.fit_train()
    y_test = model_instance.predict(df_alb_test[['date']])
    model_instance.plot_results(start_date='2019-01-01', end_date='2020-07-01')
