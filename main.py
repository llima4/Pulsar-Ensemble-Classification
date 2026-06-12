import pandas as pd

from src.data_loader import (
    load_htru1,
    load_htru2,
    split_features_target
)

from src.feature_analysis import (
    feature_summary,
    compute_vif,
    target_correlations,
    select_low_vif_features,
    select_top_importance_features,
    select_top_importance_low_vif_features
)

from src.modeling import (
    train_xgboost,
    train_random_forest,
    within_dataset_experiment,
    evaluate_model
)

from src.transfer_learning import (
    cross_dataset_experiment,
    run_transfer_feature_experiments
)

from src.visualization import (
    plot_transfer_metrics
)

import src.config

import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["xgboost","rf"],
        default="xgboost"
    )
    args = parser.parse_args()

    
    print("="*50)
    print("Loading datasets...")
    print("="*50)

    htru1 = load_htru1()
    htru2 = load_htru2()

    print("\nRunning feature analysis...\n")

    summary_h1 = feature_summary(
        htru1,
        model_type=args.model
    )

    summary_h2 = feature_summary(
        htru2,
        model_type=args.model
    )

    print("\nSaving feature summary report...\n")

    summary_h1.to_csv(
        "results/tables/htru1_feature_summary.csv"
    )

    summary_h2.to_csv(
        "results/tables/htru2_feature_summary.csv"
    )

    print("\nRunning within-dataset experiments...\n")

    model_h1, metrics_h1 = (
        within_dataset_experiment(htru1, model_type=args.model)
    )

    model_h2, metrics_h2 = (
        within_dataset_experiment(htru2, model_type=args.model)
    )

    print("="*50)
    print("Performance for within-dataset training:\n")
    results_within = pd.DataFrame(
        [metrics_h1, metrics_h2],
        index=["HTRU1","HTRU2"]
    )

    print(results_within.round(4))
    print("="*50)
    
    results_within.to_csv(
        "results/tables/within_dataset_performance.csv"
    )

    print("\nRunning transfer experiments...\n")

    results_12 = (
        run_transfer_feature_experiments(
            htru1,
            htru2,
            summary_h1,
            "HTRU1",
            "HTRU2",
            target="class",
            top_k=5,
            vif_threshold=10,
            model_type=args.model
        )
    )

    print("="*75)
    print(results_12)
    print("="*75)


    results_12.to_csv(
        "results/tables/htru12_transfer_performance.csv"
    )
    
    results_21 = (
        run_transfer_feature_experiments(
            htru2,
            htru1,
            summary_h2,
            "HTRU2",
            "HTRU1",
            target="class",
            top_k=5,
            vif_threshold=10,
            model_type=args.model
        )
    )

    print("\n")
    print("="*75)
    print(results_21)
    print("="*75)

    results_21.to_csv(
        "results/tables/htru21_transfer_performance.csv"
    )

    print("\nSaving model performance figures...\n")

    plot_transfer_metrics(
        results_12,
        "HTRU1 → HTRU2 Transfer Performance",
        save_path="results/figures/htru1_to_htru2.png",
        show=False
    )

    plot_transfer_metrics(
        results_21,
        "HTRU2 → HTRU1 Transfer Performance",
        save_path="results/figures/htru2_to_htru1.png",
        show=False
    )

if __name__ == "__main__":
    main()
