import pandas as pd
import numpy as np


def create_causal_regressors(data_full, data_date, date_column, target_column):
    """
    Function that re-calculates autoregressive variables at every forecast iteration.

    :param data_full : pd.DataFrame
        Dataset with dates and values of the target value for the training set and the predicted values of the test set
    :param data_date : pd.DataFrame
        Dataset with test time series for a particular date
    :param date_column: str
        String indicating the column that shows the date
    :param target_column: str
        String indicating the column that is the target
    :return data_date : pd.DataFrame
        Dataset with test time series for a particular date with the corrected autoregressive features
    """
    # Save column order
    col_order = list(data_date.columns)

    # Concat data_full and data_date to calculate regressors
    data_date_target = data_date[[date_column, target_column]].copy()
    data_date_target[target_column] = np.nan
    data_full_date = pd.concat([data_full, data_date_target], ignore_index=True)

    # Check the lags that exist as regressors
    auto_regressors = []
    lags = [1, 2, 3, 7]
    for lag in lags:
        data_full_date[f'lag_{lag}'] = data_full_date[target_column].shift(lag).bfill()
        auto_regressors.append(f'lag_{lag}')

    # Check the moving averages that are created
    moving_average_window = [3, 7]
    moving_average_window.sort()

    for window in moving_average_window:
        data_full_date[f'AVG_{window}'] = data_full_date[target_column].rolling(window=window, min_periods=1).mean()
        data_full_date[f'AVG_{window}'] = data_full_date[f'AVG_{window}'].shift(1).bfill()
        auto_regressors.append(f'AVG_{window}')

    # Drop auto regressors for data_date
    data_date.drop(columns=auto_regressors, inplace=True)

    # Merge new regressors
    data_full_date = data_full_date[[date_column] + auto_regressors]
    data_date = pd.merge(data_date, data_full_date, how='left', on=[date_column])
    data_date = data_date[col_order]
    return data_date


def forecast_causal(data_train, data_test, date_column, target_column, regressors, model):
    """
    Creates a forecast in a loop fashion for every data point in the dataset. For the autoregressive, we re-calculate
    them at every iteration. For the external regressors, we use the reported values

    :param target_column: str
        String indicating the column that is the target
    :param data_train : pd.DataFrame
        Dataset with training time series.
    :param data_test : pd.DataFrame
        Dataset with test time series.
    :param date_column: str
        String indicating the column that shows the date
    :param regressors : list
        Regressors, depending on the time series to add
    :param model : lightgbm.sklearn.LGBMRegressor
        LightGBM model defined by "params" and trained with the given time series.
    :return pred : np.ndarray
        Forecast of the test set
    """
    # Create an auxiliary dataframe with all the history of the target column and will populate the forecasts
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    data_full = data_train[[date_column, target_column]]

    pred = []

    # Do a forecast in a loop fashion
    for j in range(len(data_test)):
        date_i = data_test[date_column][j]
        # Filter data for that particular date
        data_date = data_test[data_test[date_column] == date_i]

        # Create new regressors and predict
        data_date = create_causal_regressors(data_full.copy(), data_date, date_column, target_column)
        pred_i = model.predict(data_date[regressors])[0]

        # Append predictions
        pred.append(pred_i)
        pred_date = pd.DataFrame(data={
            date_column: [date_i],
            target_column: [pred_i]
        })
        data_full = pd.concat([data_full, pred_date], ignore_index=True)

    pred = np.array(pred)

    return pred
