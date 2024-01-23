import seaborn as sns
import matplotlib.pyplot as plt


def plot_df(df, x_col, y_cols, min_date=None, max_date=None):

    # Filter dates
    df = df.copy()
    if min_date is not None:
        df = df[df[x_col] >= min_date]
    if max_date is not None:
        df = df[df[x_col] <= max_date]

    fig, ax = plt.subplots(figsize=(20, 6))

    # Create a seaborn plot
    for y_col in y_cols:
        sns.lineplot(x=x_col, y=y_col, data=df, label=y_col, ax=ax)

    plt.legend()
    plt.show()
