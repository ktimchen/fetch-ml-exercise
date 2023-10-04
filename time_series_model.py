import numpy as np
import pandas as pd


def clean_receipts(df):
    """Clean the data_daily table and aggregate up to a monthly level"""
    return (
        df.rename(columns={"# Date": "dt", "Receipt_Count": "count"})
        .assign(dt=pd.to_datetime(df["# Date"], format="%Y-%m-%d"))
        .set_index("dt")["count"]
        .resample("M")
        .sum()
    )


class TSModel:
    """Fit and predict linear regression on time series"""

    def __init__(self, series):
        self.series = series
        self.a, self.b = None, None

    def fit(self):
        """Returns regression coefficients for series"""
        coeffs = np.polyfit(range(self.series.size), self.series.values, deg=1)
        self.a, self.b = coeffs[0], coeffs[1]

    def predict(self, num_steps=12):
        """Forecast time series num_steps forward"""
        forecast_index = pd.date_range(
            start=self.series.index[-1],
            periods=num_steps + 1,
            freq="M",
            inclusive="right",
        )
        forecast_vals = (
            self.a * np.arange(self.series.size, self.series.size + num_steps, step=1)
            + self.b
        )
        return pd.Series(forecast_vals, index=forecast_index)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == "__main__":
    # use to print out last 3-month accuracy
    num_receipts = pd.read_csv("data_daily.csv")
    num_receipts = clean_receipts(num_receipts)
    model = TSModel(num_receipts[:-3])
    model.fit()
    mape = mean_absolute_percentage_error(num_receipts[-3:], model.predict(3))
    print(f"MAPE error on a test set of Oct, Nov and Dec 2021 is {mape:.2f}%")
