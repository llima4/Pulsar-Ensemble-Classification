import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import src.config

# Feature distributions
def plot_feature_distributions(df, target):
    features = [c for c in df.columns if c != target]

    ncols = 3
    nrows = int(np.ceil(len(features)/ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15,4*nrows)
    )

    axes = axes.flatten()

    for ax, feature in zip(axes, features):

        sns.histplot(
            data=df,
            x=feature,
            hue=target,
            stat="density",
            common_norm=False,
            ax=ax
        )

    plt.tight_layout()


# Feature corrleations with class label
def plot_target_correlations(df, target):

    corr = (
        df.corr(numeric_only=True)[target]
        .drop(target)
        .sort_values(key=np.abs, ascending=False)
    )

    plt.figure(figsize=(10,6))

    corr.sort_values().plot.barh()

    plt.title("Feature Correlation with Class Label")
    plt.show()


# Feature correlation heatmap
def plot_correlation_heatmap(df, target):

    corr = df.drop(columns=[target]).corr()

    plt.figure(figsize=(12,10))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Matrix")
    plt.show()


# Feature importances
def feature_importances(importance):
    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    importance.plot.bar()
    plt.title("Feature Importances")
    plt.show()


# Visualize cross-dataset model results
def plot_transfer_metrics(
    results,
    title
):

    metrics = [
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]

    ax = (
        results[metrics]
        .plot(
            kind="bar",
            figsize=(10,6)
        )
    )

    ax.set_ylim(0,1)

    plt.title(title)

    plt.ylabel("Score")

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.show()
