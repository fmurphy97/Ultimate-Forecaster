import pandas as pd


def split_series(df, start_test_date, end_test_date, date_col_name="date"):
    """
    Splits a time series into train and test sets according to the given date boundaries.

    Note that all records before the start of the test set are considered to be part of the train set. Furthermore, the
    end of the train set and start of the test set are assumed to be contiguous.

    Parameters
    __________
        df (pd.DataFrame): Dataset with the time series to split.
        start_test_date (str): Start of test set (included).
        end_test_date (str): End of test set (included).

    Returns
    ________
        train_df (pd.DataFrame): Train set DataFrame.
        test_df (pd.DataFrame): Test set DataFrame.
    """

    # Splitting dataset
    train_df = df[df[date_col_name] < pd.to_datetime(start_test_date)]
    test_df = df[(df[date_col_name] >= pd.to_datetime(start_test_date)) &
                 (df[date_col_name] <= pd.to_datetime(end_test_date))]

    return train_df, test_df
