from pathlib import Path
import pandas as pd
import src.config

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DATA_DIR = PROJECT_ROOT / "data"


def load_htru1():

    return pd.read_csv(
        DATA_DIR / "HTRU1_Combined_Features.csv"
    )


def load_htru2():

    return pd.read_csv(
        DATA_DIR / "HTRU2_Combined_Features.csv"
    )


def split_features_target(
    df,
    target
):

    X = df.drop(columns=[target])

    y = df[target]

    return X, y
