from forecasting_models.forecast_model import ForecastModel
import pandas as pd
from lightgbm.sklearn import LGBMRegressor


class LightGBM(ForecastModel):

    def __init__(self, df, x_col_name, y_col_name, external_regressors=None, **model_params):
        super().__init__(df, x_col_name, y_col_name, external_regressors)

        self.model_params = model_params

        self.model_name = f"LightGBM"  # TODO: add as many parameters as desired

    def fit_train(self):
        self.model = LGBMRegressor(**self.model_params)
        self.model.fit(self.x_train, self.y_train)

    def predict(self, x_test):
        self.df = pd.concat([self.df, x_test[[self.x_col_name]]], axis=0, ignore_index=True)

        self.df[self.model_name] = self.model.predict(x_test)

        values_to_forecast = len(x_test)
        return self.df[[*self.all_regressors, self.model_name]][-values_to_forecast:]


df_alb = pd.read_csv("../data/peajes_alberdi_training_preproc.csv")
df_alb['date'] = pd.to_datetime(df_alb['fecha'], format='%Y-%m-%d')

df_alb_train = df_alb.copy()[df_alb['date'] < '2019-07-01']
df_alb_test = df_alb.copy()[df_alb['date'] >= '2019-07-01']

model_instance = LightGBM(df=df_alb_train.copy(), x_col_name='date', y_col_name='cantidad_pasos')
model_instance.fit_train()
y_test = model_instance.predict(df_alb_test.copy()[['date']])
model_instance.plot_results(start_date='2016-08-01', end_date='2017-08-01')
