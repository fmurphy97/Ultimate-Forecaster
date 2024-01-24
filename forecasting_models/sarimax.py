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

        self.df.set_index(self.x_col_name, inplace=True)
        self.order = order
        self.seasonal_order = seasonal_order

        model_text = "ARIMA"

        if external_regressors is not None:
            model_text = model_text + "X"
        if seasonal_order != (0, 0, 0, 0):
            model_text = "S" + model_text
            self.model_name = f"{model_text}_{order}_{seasonal_order}"
        else:
            self.model_name = f"{model_text}_{order}"

    def fit_train(self):
        if self.external_regressors is not None:
            exogenous_features_df = self.df[[self.external_regressors]]
        else:
            exogenous_features_df = None
        self.model = ARIMA(self.df[self.y_col_name], exog=exogenous_features_df,
                           order=self.order, seasonal_order=self.seasonal_order).fit()

    def predict(self, x_test):
        x_test.set_index(self.x_col_name, inplace=True)

        # BUG: predict method not working
        if self.external_regressors is not None:
            exogenous_features_df = x_test[[self.external_regressors]]
        else:
            exogenous_features_df = None

        fit = list(self.model.predict())
        pred = list(self.model.predict(
            start=x_test.index.min(),
            end=x_test.index.max(),
            exog=exogenous_features_df))
        return fit, pred
