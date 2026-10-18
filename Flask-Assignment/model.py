"""Load the trained beer alcohol regression model and make predictions."""

from pathlib import Path
import pickle

import pandas as pd


MODEL_PATH = Path(__file__).resolve().with_name("model.pkl")
FEATURES = [
    "country",
    "beer_servings",
    "spirit_servings",
    "wine_servings",
    "continent",
]


def load_model():
    """Load the fitted scikit-learn pipeline stored beside this file."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} is missing. Run the cells in model_selection.ipynb first."
        )
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def predict_alcohol(model, country, beer_servings, spirit_servings, wine_servings, continent):
    """Return a non-negative prediction in litres of pure alcohol."""
    row = pd.DataFrame(
        [{
            "country": country,
            "beer_servings": beer_servings,
            "spirit_servings": spirit_servings,
            "wine_servings": wine_servings,
            "continent": continent,
        }],
        columns=FEATURES,
    )
    return max(0.0, float(model.predict(row)[0]))
