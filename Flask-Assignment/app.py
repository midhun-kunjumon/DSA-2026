"""Flask interface for the beer servings regression model."""

from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from model import FEATURES, load_model, predict_alcohol


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "beer-servings.csv"
app = Flask(__name__)
model = load_model()
data = pd.read_csv(DATA_PATH)
data = data.dropna(subset=FEATURES + ["total_litres_of_pure_alcohol"])
continents = sorted(data["continent"].dropna().unique().tolist())
countries = sorted(data["country"].dropna().unique().tolist())
summary = {
    "countries": int(data["country"].nunique()),
    "continents": int(data["continent"].nunique()),
    "average_alcohol": float(data["total_litres_of_pure_alcohol"].mean()),
    "average_beer": float(data["beer_servings"].mean()),
}
chart_stats = (
    data.groupby("continent")["total_litres_of_pure_alcohol"]
    .mean()
    .sort_values(ascending=False)
    .round(1)
    .to_dict()
)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = {
        "country": countries[0] if countries else "",
        "continent": continents[0] if continents else "",
        "beer_servings": "0",
        "spirit_servings": "0",
        "wine_servings": "0",
    }

    if request.method == "POST":
        values.update({key: request.form.get(key, "").strip() for key in values})
        try:
            if values["country"] not in countries:
                raise ValueError("Choose a country from the list.")
            if values["continent"] not in continents:
                raise ValueError("Choose a continent from the list.")
            servings = []
            for key in ("beer_servings", "spirit_servings", "wine_servings"):
                number = int(values[key])
                if number < 0 or number > 1000:
                    raise ValueError("Servings must be between 0 and 1,000.")
                servings.append(number)
            prediction = predict_alcohol(model, values["country"], *servings, values["continent"])
        except (TypeError, ValueError):
            error = "Enter valid whole-number servings from 0 to 1,000 and choose both categories."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=values,
        countries=countries,
        continents=continents,
        summary=summary,
        chart_stats=chart_stats,
    )


if __name__ == "__main__":
    app.run(debug=True)
