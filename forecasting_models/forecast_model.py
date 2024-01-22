import matplotlib.pyplot as plt


class ForecastModel:
    def __init__(self, df, x_col_name="date", y_col_name="y", external_regressors=None):
        """
        :param df: (pd.Dataframe) Dataset with training time series.
        :param x_col_name: (str, optional) Name of the column that contains the dates. Defaults to "date".
        :param y_col_name: (str, optional) Name of the column containing the dependent variable. Defaults to "y".
        :param external_regressors: (list) List of strings of column names of the exogenous variables to use.
        """
        self.model = None
        self.model_name = ""
        self.df = df

        self.x_col_name = x_col_name
        self.y_col_name = y_col_name

        self.x_train = self.df[self.x_col_name]
        self.y_train = self.df[self.y_col_name]
        if external_regressors is not None:
            self.exogenous_train = self.df[external_regressors]
        else:
            self.exogenous_train = None

    def fit_train(self):
        """
        Fits the forecasting model to the training data.
        """
        pass

    def predict(self, x_test):
        """
        Predicts target variable values for the given input dates

        :param x_test: pd.DataFrame
            Input dates for which predictions are to be made.
        :return: pd.DataFrame
            Predicted values for the target variable.
        """
        pass

    def plot_results(self, start_date: str, end_date: str) -> None:
        # Filter by date
        df_filtered = self.df[(self.df[self.x_col_name] >= start_date) & (self.df[self.x_col_name] <= end_date)]

        # Create the image
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(df_filtered[self.x_col_name], df_filtered['cantidad_pasos'], label='Historical')
        ax.plot(df_filtered[self.x_col_name], df_filtered[self.model_name], label=self.model_name)
        ax.legend()
        ax.tick_params(labelrotation=90)
        plt.tight_layout()
        plt.show()
