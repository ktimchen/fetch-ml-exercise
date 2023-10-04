import base64
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request

from time_series_model import TSModel, clean_receipts

PLACEHOLDER_URL = "https://via.placeholder.com/400/?text="

app = Flask(__name__)

# fit the forecast model on the training data
num_receipts = pd.read_csv("data_daily.csv")
num_receipts = clean_receipts(num_receipts)
model = TSModel(num_receipts)
model.fit()


def plot_forecast(train_series, forecast_series):
    """Plot two time series, train and forecast, on one graph. Return base64 encoded png"""

    # turn on the static backend for matplotlib
    matplotlib.use("Agg")

    train_series.plot(style="-o", label="2021 data", xlabel="")
    forecast_series.plot(style="-o", label="forecast", xlabel="")

    # aesthetics settings
    plt.grid(True, axis="y")
    plt.legend()
    plt.ticklabel_format(axis="y", style="plain")
    ay = plt.gca().get_yaxis()
    ay.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    # base64 encode the graph
    with BytesIO() as f:
        plt.savefig(f, bbox_inches="tight", format="png")
        plt.close()
        base64_png = base64.b64encode(f.getvalue()).decode()
    return base64_png


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/forecast", methods=["POST"])
def predict():
    """
    1. Capture the forecast horizon provided by user (integer)
    2. Make predictions
    3. Render graph and predictions (as a table)
    """
    num_steps = next(request.form.values())
    try:
        num_steps = int(num_steps)
    except ValueError as e:
        return render_template(
            "index.html", placeholder_url=PLACEHOLDER_URL + "use+integers!"
        )
    if not 1 <= num_steps <= 400:
        return render_template(
            "index.html", placeholder_url=PLACEHOLDER_URL + "use+int+between+1+and+400"
        )

    # make the prediction num_steps forward
    forecast = model.predict(num_steps)
    base64_png = plot_forecast(num_receipts, forecast)

    # pass forecast as a dict to render
    forecast.index = forecast.index.strftime("%b %Y")
    forecast = forecast.round().astype(int).apply(lambda x: "{:,}".format(x))
    forecast_table = forecast.to_dict().items()

    return render_template(
        "index.html", img_to_display_base64=base64_png, table=forecast_table
    )


if __name__ == "__main__":
    # app.run(host="0.0.0.0")
    app.run()
