"""
to be done by person 2

Feature engineering, feature understanding and feature reduction.

Planned contents:
- build_feature_table(): one row per clip x ASR model with features that do NOT need the true transcript
  (duration, SNR, predicted word count, speaking rate, confidence, model) and the targets (wer, hard clip)
- drop_correlated(): remove one feature of each highly correlated pair
- analyse_features(): correlation matrix, PCA explained variance, random-forest importance
- main(): save outputs/features.csv, feature_analysis.txt and plots
Run with: python -m src.features
"""

# to be done by person 2
