# Pulsar-Ensemble-Classification

## Project Goal
1. Build robust ensemble learning models (XGBoost and random forest) for classifying pulsar candidates.
2. Investigate feature selection and cross-dataset generalization using the HTRU1 and HTRU2 datasets.

## Datasets

This project uses data from the High Time Resolution Universe Survey. The source for the HTRU 1 and HTRU 2 datasets can be found at:

https://github.com/scienceguyrob/PulsarFeatureLab/

References:
[1] R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Submitted to MNRAS.

[2] R. J. Lyon et al., "Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach", Submitted to Monthly Notices of the Royal Astronomical Society.

[3] D. Thornton, "The High Time Resolution Radio Sky", PhD thesis, University of Manchester, Jodrell Bank Centre for Astrophysics School of Physics and Astronomy, 2013.

## Research Questions

1. Which features are most predictive?
2. Which features are redundant?
3. Does feature reduction improve cross-dataset performance?

## Methods

- Feature Distribution Analysis
- Feature Correlation Analysis
- Variance Inflation Factor
- XGBoost Feature Importance
- Random Forest Feature Importance
- Cross-Dataset Testing

