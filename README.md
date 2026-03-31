# Predicting Fasciolosis in Uruguayan Bovine Carcasses: An End-to-End Machine Learning Study

## Description
This repository implements a supervised machine learning pipeline to predict fasciolosis (liver fluke infection) in bovine carcasses processed at a major Uruguayan abattoir in 2016. Using routinely collected post-slaughter data, the model leverages Light Gradient Boosting Machine (LightGBM) with Bayesian hyperparameter optimization, feature engineering, and rigorous evaluation. The approach prioritizes sensitivity (recall) for disease detection, using the three most informative features: carcass weight, dental age, and fat score. All pipeline components—including data preprocessing, imputation, feature ranking, model tuning, and performance assessment—are designed for full reproducibility and operational relevance in abattoir surveillance workflows.

## Overview
- **Target**: Binary indicator of fasciolosis presence/absence in bovine carcasses.
- **Key predictive features**:
  - Carcass weight (continuous)
  - Dental age (`age_teeth`; ordinal)
  - Fat score (ordinal)
- **Additional optional variables** explored but not used in final model:
  - Productive purpose (categorical)
  - Carcass conformation score (numeric)
  - Month (derived from slaughter date)
- **Model**: LightGBM classifier with Bayesian hyperparameter tuning (maximizing AUC-ROC).
- **Training & Evaluation**:
  - Stratified 5-fold cross-validation on training data for hyperparameter tuning.
  - Held-out test set evaluation on multiple metrics including AUC-ROC, recall, precision, F1-score, accuracy, and confusion matrix.
- **Operating point**: Sensitivity-oriented (high recall) threshold to reduce missed infected carcasses at the cost of increased false positives.

## Data
- Source: Publicly available abattoir dataset [[1](https://data.mendeley.com/datasets/3jnn876my4/3)].
- Sampling frame: Bovine carcass-level observations from one of Uruguay's largest exporters during 2016.
- Target variable: Fasciolosis presence (`fasciola` binary indicator).
- Features:  
  | Feature                | Type      | Description                                   |
  |------------------------|-----------|-----------------------------------------------|
  | `carcass_weight`       | Continuous| Post-slaughter carcass weight                  |
  | `age_teeth`            | Ordinal   | Dental age classification                      |
  | `fat_score`            | Ordinal   | Fat cover score                                |
  | `carcass_conformation_score` | Numeric    | Conformation score (not in final model)          |
  | `productive_purpose`   | Categorical | Production purpose (one-hot encoded, optional) |
  | `month`                | Ordinal   | Derived from slaughter date (1-12)            |

- Preprocessing steps:
  - Standardized variable names.
  - Missing values imputed with median (numerical) or mode (categorical).
  - One-hot encoding for categorical variables with unknown category handling.

## Methods
1. **Data preparation and preprocessing**  
   - Clean and standardize column names.
   - Derive month from date field.
   - Impute missing values.
   - Encode categorical variables (one-hot for `productive_purpose` when included).

2. **Feature ranking**  
   - Random Forest classifier-based feature importance ranking on training data.
   - Group one-hot encoded variables back to original feature level.
   - Select top three features (`carcass_weight`, `age_teeth`, `fat_score`) to ensure parsimony and interpretability.

3. **Modeling and hyperparameter tuning**  
   - Use LightGBM binary classifier with balanced class weights.
   - Bayesian optimization over:
     - `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`
   - Optimization target: maximize mean AUC-ROC via 5-fold stratified cross-validation on training set.
   - Early stopping employed during cross-validation.
   - Final model trained on full training data restricted to top 3 features with best hyperparameters.

4. **Evaluation**  
   - Held-out test set evaluation: AUC-ROC, accuracy, precision, recall, F1-score.
   - Confusion matrix analysis and class-wise metrics for interpretability.
   - Emphasis on recall (sensitivity) to detect infected carcasses.

## Results
- **Feature importance (Random Forest baseline)**:
  - `carcass_weight`: 0.876
  - `age_teeth`: 0.082
  - `fat_score`: 0.017
- **Hyperparameter tuning**:
  - Best AUC-ROC (cross-validation): ~0.6510
  - Optimal hyperparameters:
    - `num_leaves`=16
    - `learning_rate`≈0.2534
    - `n_estimators`=767
    - `min_child_samples`=63
    - `subsample`≈0.6889
    - `colsample_bytree`≈0.7264
    - `reg_lambda`≈1.4116
    - `reg_alpha`≈2.0948
    - `class_weight`=balanced
- **Test set performance**:
  - AUC-ROC: 0.6519
  - Accuracy: 0.5822
  - Precision (infected class): 0.4451
  - Recall (infected class): 0.7263
  - F1-score: 0.5519
  - Confusion matrix:
    - True Negative (TN): 1959
    - False Positive (FP): 1935
    - False Negative (FN): 585
    - True Positive (TP): 1552
- Interpretation: Model favors recall (sensitivity), suitable for minimizing missed infections in surveillance contexts.

## Discussion and Future Work
- Dominant signal from carcass weight aligns with known biological impacts of fasciolosis on animal physiology.
- Age and fat score provide supplemental but weaker signals.
- The model is best suited for abattoir-stage detection, where carcass physiology is apparent.
- Pre-slaughter or herd-level prediction will require upstream environmental and management data to avoid label leakage.
- Potential improvements:
  - Expanding feature set (multi-year temporal data, farm/environment covariates).
  - Alternative interpretability approaches (SHAP values, permutation importance).
  - Investigate other learners (XGBoost, CatBoost) or ensemble methods.
  - Implement probability calibration and threshold tuning for operational cost-benefit optimization.
  - Incorporate precision–recall and cost-sensitive metrics aligned with real-world trade-offs.


---

## License
This project is released under the MIT License.

---

## Questions?
For questions, suggestions, or collaboration, please open an issue in this repository or contact me at jprmaulion[at]gmail[dot]com.
