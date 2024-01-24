from statsmodels.tsa.arima.model import ARIMA
from forecasting_models.forecast_model import ForecastModel


class Sarimax(ForecastModel):

    def __init__(self, df, x_col_name, y_col_name, external_regressors=None, order=(0, 0, 0),
                 seasonal_order=(0, 0, 0, 0)):
        """
        :param order: integration models: ARIMA(p, d, q)
        :param seasonal_order: seasonal models: SARIMA(P, D, Q, s)
        """
        super().__init__(df=df, x_col_name=x_col_name, y_col_name=y_col_name, external_regressors=external_regressors)

        self.df = self.df.set_index(self.x_col_name, inplace=True)
        self.order = order
        self.seasonal_order = seasonal_order

        self.model_name = (f"ARIMA"
                           f"_{order}"
                           f"_{seasonal_order}")

    def fit_train(self):
        if self.external_regressors is not None:
            exogenous_features_df = self.df[[self.external_regressors]]
        else:
            exogenous_features_df = None
        self.model = ARIMA(self.y_train, exog=exogenous_features_df, dates=self.df[self.x_col_name],
                           order=self.order, seasonal_order=self.seasonal_order).fit()

    def predict(self, x_test):
        # BUG: predict method not working
        if self.external_regressors is not None:
            exogenous_features_df = x_test[[self.external_regressors]]
        else:
            exogenous_features_df = None

        fit = list(self.model.predict())
        pred = list(self.model.predict(
            start=x_test[self.x_col_name].min().strftime('%Y-%m-%d'),
            end=x_test[self.x_col_name].max().strftime('%Y-%m-%d'),
            exog=exogenous_features_df))
        return fit, pred


if __name__ == "__main__":
    import pandas as pd

    df_alb = pd.read_csv("../data/peajes_alberdi_training_preproc.csv")
    df_alb['date'] = pd.to_datetime(df_alb['fecha'], format='%Y-%m-%d')
    df_alb_train = df_alb.copy()[df_alb['date'] < '2019-07-01']
    df_alb_test = df_alb.copy()[df_alb['date'] >= '2019-07-01']
    x_feature_name = "date"
    y_feature_name = "cantidad_pasos"

    model_class = Sarimax

    model_instance = model_class(df=df_alb_train.copy(), x_col_name=x_feature_name, y_col_name=y_feature_name,
                                 order=(7, 0, 3))
    model_instance.fit_train()
    fitted_data, predicted_data = model_instance.predict(df_alb_test.copy()[[x_feature_name]])
    df_alb[model_instance.model_name] = fitted_data + predicted_data
