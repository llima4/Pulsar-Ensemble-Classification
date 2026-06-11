import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)

import src.config

from src.data_loader import *
from src.modeling import (
    train_xgboost,
    train_random_forest
    )

# VIF metric for measuring multicollinearity
def compute_vif(df, target):

    X = df.drop(columns=[target])

    vif_df = pd.DataFrame()

    vif_df["Feature"] = X.columns

    vif_df["VIF"] = [
        variance_inflation_factor(
            X.values,
            i
        )
        for i in range(X.shape[1])
    ]

    return vif_df.sort_values(
        "VIF",
        ascending=False
    )


# Feature correlations with class label
def target_correlations(df, target="class"):

    corr = (
        df.corr(numeric_only=True)[target]
        .drop(target)
    )

    return corr


def feature_summary(
    df,
    model_type="xgboost",
    target="class"
):

    X, y = split_features_target(
        df,
        target
    )

    if model_type == "xgboost":

        model = train_xgboost(
            X,
            y
        )

    elif model_type == "rf":

        model = train_random_forest(
            X,
            y
        )

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
        name="Importance"
    )

    vif = compute_vif(
        df,
        target
    )

    corr_target = target_correlations(
        df,
        target
    )

    feature_summary = pd.DataFrame({
        "Target_Corr": corr_target,
        "Importance": importance
    })

    feature_summary["Abs_Target_Corr"] = (
        feature_summary["Target_Corr"].abs()
    )

    feature_summary = (
        feature_summary
        .merge(
            vif,
            left_index=True,
            right_on="Feature"
        )
        .set_index("Feature")
    )

    feature_summary["Importance_Rank"] = (
        feature_summary["Importance"]
        .rank(ascending=False)
    )

    feature_summary["Corr_Rank"] = (
        feature_summary["Abs_Target_Corr"]
        .rank(ascending=False)
    )

    return feature_summary

# Collect features based on given metric thresholds
def select_low_vif_features(
    feature_summary,
    vif_threshold=10
):

    return list(
        feature_summary[
            feature_summary["VIF"] < vif_threshold
        ].index
    )


def select_top_importance_features(
    feature_summary,
    top_k=5
):

    return list(
        feature_summary
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_k)
        .index
    )


def select_top_importance_low_vif_features(
    feature_summary,
    top_k=5,
    vif_threshold=10
):

    filtered = feature_summary[
        feature_summary["VIF"] < vif_threshold
    ]

    return list(
        filtered
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_k)
        .index
    )

