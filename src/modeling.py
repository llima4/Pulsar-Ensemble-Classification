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

import src.config

# Train XGBoost model
def train_xgboost(X_train, y_train):

    model = XGBClassifier(
        eval_metric="logloss",
        random_state
    )

    model.fit(X_train, y_train)

    return model


# Train random forest model
def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=500,
        random_state
    )

    model.fit(X_train, y_train)

    return model


# Collect Model Performance Metrics 
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }


#### EXPERIMENT A ####
## Compare feature importances and multicollinearity across datasets

# Train and test data from the same dataset 
def within_dataset_experiment(
    df,
    model_type="xgboost"
):

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            stratify=y,
            test_size=0.2,
            random_state
        )
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

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
        name="Importance"
    )

    return model, metrics, importance
