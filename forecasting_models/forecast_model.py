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

        # TODO: add these params
        # self.train_start_end_dates = []
        # self.model_periods = "D", "W", "M", etc

        self.all_regressors = [x_col_name]
        self.external_regressors = external_regressors
        if external_regressors is not None:
            self.all_regressors += external_regressors
        self.x_train = self.df[self.all_regressors]
        self.y_train = self.df[self.y_col_name]

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
