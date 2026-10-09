"""
============================================================
Standard Predictive Modeling Methodology
============================================================

This document defines the universal predictive modeling workflow
that EndForecast implements. It is derived from decades of
industry best practices in data science, machine learning,
and statistical forecasting.

The workflow is **task-agnostic**: it applies identically to
classification, regression, and time series problems.

============================================================
PHASE FLOW (10 phases, executed sequentially)
============================================================

0. BUSINESS UNDERSTANDING
   - Define the prediction goal in business terms
   - Translate to ML task type (classification/regression/time series)
   - Establish success criteria and acceptable performance thresholds
   - Identify constraints: temporal, computational, regulatory, domain

1. DATA UNDERSTANDING
   - Schema inference: column types, roles (target/feature/time/id)
   - Statistical profiling: distributions, missing ratios, cardinality
   - Data quality assessment: duplicates, outliers, inconsistencies
   - Temporal structure analysis (for time series): frequency, gaps

2. DATA PREPARATION
   - Missing value treatment: imputation or deletion based on pattern
   - Outlier handling: capping, transformation, or flagging
   - Encoding: one-hot (low-cardinality), target-encoding (high-card),
     label-encoding (ordinal)
   - Scaling: standard (normal-dist), min-max (bounded), robust (outliers)
   - Class balancing: SMOTE, class weights, or cost-sensitive loss (classification)

3. EXPLORATORY DATA ANALYSIS (EDA) + FEATURE ENGINEERING
   - Distribution analysis: skew, kurtosis, multimodality
   - Correlation structure: Pearson, Spearman, mutual information
   - Temporal patterns: seasonality, trend, cycles, change points (time series)
   - Feature construction: lags, rolling statistics, calendar features,
     interaction terms, domain-specific features
   - Dimensionality assessment: PCA, variance explained, feature selection

4. BASELINE ESTABLISHMENT
   - Trivial baselines: mean, median, mode, last-value
   - Seasonal naive (time series): copy same hour/day from previous cycle
   - Linear baseline: linear/logistic regression with minimal features
   - Quick tree baseline: LightGBM/RandomForest with defaults
   ---- CRITICAL RULE ----
   NEVER skip baselines. A complex model that barely beats a naive
   baseline is likely overfit or the data has no predictive signal.

5. MODEL EXPERIMENTATION
   - Systematic cross-family comparison (not random exploration)
   - At least 3 model families tested before any hyperparameter tuning
   - Proper CV strategy: temporal for time series, stratified for
     imbalanced classification, standard K-fold otherwise
   - Hyperparameter optimization only after best family identified

6. EVALUATION & ERROR ANALYSIS
   - Multi-metric evaluation (never just one metric)
   - Residual analysis: distribution, autocorrelation, heteroscedasticity
   - Segmented error analysis: by time period, value range, class
   - Confusion matrix with per-class metrics (classification)
   - Calibration analysis: reliability diagrams (classification)
   - Best-worst analysis: identify easiest and hardest prediction cases

7. ITERATIVE REFINEMENT
   - Diagnose failure patterns from error analysis
   - Targeted fixes: each adjustment addresses a specific diagnosed issue
   - Avoid shotgun tuning: don't change multiple things at once
   - Track improvement vs. baseline (not vs. previous iteration)
   - Convergence check: diminishing returns signals completion

8. INTERPRETATION & EXPLAINABILITY
   - Global feature importance: SHAP, permutation, gain
   - Local explanations: SHAP waterfall for individual predictions
   - Partial dependence: how each feature affects predictions
   - Business-level translation: what does this mean for decisions?
   ---- RULE ----
   An unexplainable model in a high-stakes domain is unacceptable.

9. DEPLOYMENT
   - Model serialization: weights + preprocessing pipeline as one unit
   - Input validation: schema checking, range checking, type checking
   - API/service layer: REST, gRPC, or embeddable code
   - Monitoring hooks: log predictions, inputs, and latencies
   - Version tracking: model lineage, training data hash, config snapshot

10. CONTINUOUS MONITORING & ADAPTATION
    - Performance monitoring: track prediction error over time
    - Data drift detection: PSI (Population Stability Index),
      KS-test, KL divergence for input distributions
    - Concept drift detection: systematic error increase over time
    - Retraining triggers: time-based (schedule), data-volume-based,
      performance-based (error exceeds threshold)
    - A/B deployment: shadow mode → canary → full rollout
    - Rollback capability: keep previous model version active

============================================================
METHODOLOGICAL PRINCIPLES
============================================================

1. LEAKAGE PREVENTION IS NON-NEGOTIABLE
   Verify temporal boundaries BEFORE any model training.
   One undetected leak invalidates everything downstream.

2. BASELINES FIRST, COMPLEXITY LATER
   Simple baselines provide a sanity check and an improvement ceiling.
   If a complex model doesn't beat a trivial one, question the data.

3. CROSS-VALIDATION MUST MATCH PROBLEM STRUCTURE
   Temporal → time series split (no shuffle).
   Classification → stratified split.
   i.i.d. → standard K-fold.

4. MULTI-METRIC EVALUATION IS ESSENTIAL
   A single metric hides important failure modes.
   Classification: F1 + AUC + CM + calibration.
   Regression: MAE + MAPE + R² + residual analysis.
   Time series: MASE + SMAPE + directional accuracy.

5. ERROR ANALYSIS DRIVES IMPROVEMENT
   Understand WHERE and WHY models fail before trying to fix them.
   Segmented analysis reveals patterns invisible to aggregate metrics.

6. INTERPRETABILITY IS A REQUIREMENT, NOT A FEATURE
   Black-box predictions are dangerous in business/medical/financial
   applications. Always provide explanation capabilities.

7. PIPELINE REPRODUCIBILITY
   The entire workflow (data→model→deployment) must be reproducible
   from a single configuration. No manual steps, no ad-hoc fixes.

8. MODELS DEGRADE — DETECT IT
   No model stays accurate forever. Data distributions shift, concepts
   change. Continuous monitoring is not optional for production systems.

9. CONSERVATIVE REFINEMENT
   Post-hoc corrections should be applied sparingly. Each correction
   must have a clear diagnostic basis and an estimated benefit.
   "Better no refinement than wrong refinement."

10. TASK-AGNOSTIC ARCHITECTURE
    The workflow is identical for classification, regression, and
    time series. Only the specific methods and metrics change.
"""