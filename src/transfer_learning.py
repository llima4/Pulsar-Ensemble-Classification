import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.model_selection import train_test_split

from src.modeling import *
from src.feature_analysis import *
from src.data_loader import *

import src.config

#### EXPERIMENT B ####
## Analyzing model performance with cross-dataset validation

def cross_dataset_experiment(
    train_df,
    test_df,
    model_type="xgboost"
):

    X_train, y_train = (
        split_features_target(train_df)
    )

    X_test, y_test = (
        split_features_target(test_df)
    )

    if model_type == "xgboost":

        model = train_xgboost(
            X_train,
            y_train
        )

    elif model_type == "rf":

        model = train_random_forest(
            X_train,
            y_train
        )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    return model, metrics


def run_transfer_feature_experiments(
    train_df,
    test_df,
    train_summary,
    train_name,
    test_name,
    target,
    top_k,
    vif_threshold
):

    experiments = {}

    # All Features
    _, metrics = cross_dataset_experiment(
        train_df,
        test_df,
        target
    )

    experiments["All Features"] = metrics

    # Low VIF
    low_vif = select_low_vif_features(
        train_summary,
        vif_threshold
    )

    train_low_vif = train_df[
        low_vif + [target]
    ]

    test_low_vif = test_df[
        low_vif + [target]
    ]

    _, metrics = cross_dataset_experiment(
        train_low_vif,
        test_low_vif,
        target
    )

    experiments["Low VIF"] = metrics

    # Top Importance
    top_features = (
        select_top_importance_features(
            train_summary,
            top_k
        )
    )

    train_top = train_df[
        top_features + [target]
    ]

    test_top = test_df[
        top_features + [target]
    ]

    _, metrics = cross_dataset_experiment(
        train_top,
        test_top,
        target
    )

    experiments["Top Importance"] = metrics

    # Top Importance + Low VIF
    top_low_vif = (
        select_top_importance_low_vif_features(
            train_summary,
            top_k,
            vif_threshold
        )
    )

    train_top_vif = train_df[
        top_low_vif + [target]
    ]

    test_top_vif = test_df[
        top_low_vif + [target]
    ]

    _, metrics = cross_dataset_experiment(
        train_top_vif,
        test_top_vif,
        target
    )

    experiments["Top Importance + Low VIF"] = metrics

    results = pd.DataFrame(
        experiments
    ).T

    results.index.name = (
        f"{train_name} → {test_name}"
    )

    return results
