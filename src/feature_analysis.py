import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)

import src.config

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


# Dataframe of metrics for each feature
def feature_summary(importance, corr_target, vif):
    feature_summary = pd.DataFrame({
            "Target_Corr": corr_target,
            "XGB_Importance": importance
        })

        feature_summary = (
            feature_summary
            .merge(vif, left_index=True,
                   right_on="Feature")
            .set_index("Feature")
        )

        feature_summary["Importance_Rank"] = (
            feature_summary["XGB_Importance"]
            .rank(ascending=False)
        )

        feature_summary["Corr_Rank"] = (
            feature_summary["Abs_Target_Corr"]
            .rank(ascending=False)
        )


# Collect features based on given metric thresholds
def select_low_vif_features(
    feature_summary,
    vif_threshold
):

    return list(
        feature_summary[
            feature_summary["VIF"] < vif_threshold
        ].index
    )


def select_top_importance_features(
    feature_summary,
    top_k
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
    top_k,
    vif_threshold
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

