"""
to be done by person 2

Machine-learning models that predict how hard a clip is for the ASR system.

Planned contents:
- regression: predict the clip's WER (baseline, linear regression, decision tree, random forest, SVR, small neural net)
- classification: predict whether the clip is "hard" (WER above a threshold)
  (baseline, logistic regression, decision tree, random forest, SVM, small neural net)
- three feature sets: all features, correlation-reduced features, PCA (95% variance)
- group-aware train/test split so clean and noisy copies of one clip never leak across the split
- save outputs/model_results.csv, outputs/test_predictions.csv and the best models (joblib)
Run with: python -m src.train_model
"""

# to be done by person 2
