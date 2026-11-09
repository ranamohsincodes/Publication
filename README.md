# Explainable Ensemble Learning for Loan Default Prediction

An experimental framework evaluating machine learning models for loan default prediction across multiple operational dimensions: discrimination, probability calibration, explanation stability, and temporal generalization. 

This repository contains the implementation pipeline based on the research paper **"Explainable Ensemble Learning for Loan Default Prediction: An Evaluation of Probability Calibration and Explanation Stability"** (Qadir et al., 2026).

## 📌 Project Overview
Predicting loan defaults requires models that not only rank borrowers accurately but also provide reliable probability estimates and stable explanations over time. This project implements a unified evaluation framework to compare traditional and ensemble machine learning approaches using Lending Club loan records.

### Key Operational Dimensions Evaluated:
1. **Unified Evaluation:** Joint assessment of discrimination, minority-class detection, probability calibration, and explanation stability.
2. **Temporal Validation:** Utilizing a chronological split (2007–2018) with half-year test windows to evaluate real-world model degradation.
3. **Explanation Stability:** Robustness analysis of SHAP-based feature importance rankings via bootstrap refitting.
4. **Reproducible Pipeline:** Standardized data preprocessing, class imbalance handling via weights, and validation-based decision-threshold optimization.

## 📊 Models Evaluated
* **Logistic Regression** (Interpretable baseline)
* **Random Forest** (Bagging ensemble)
* **XGBoost** (Gradient boosting)
* **LightGBM** (Gradient boosting)
* **Stacking Ensemble** (Meta-learner combining out-of-fold probabilities)

## ⚙️ Methodology & Pipeline
1. **Data Preprocessing:** Outlier winsorization, median imputation, and scaling isolated cleanly within the training partition to prevent data leakage.
2. **Imbalance Handling:** Built-in class weights used within each learner to preserve the natural validation and test distributions.
3. **Probability Calibration:** Post-hoc validation-based sigmoid calibration (Platt scaling) to improve the reliability of estimated risks.
4. **Explainable AI (XAI):** Global and local feature attributions using exact `TreeSHAP` and linear SHAP metrics.
5. **Stability Assessment:** Evaluation of SHAP rank consistency and top-k Jaccard overlap over repeated bootstrap samples.

## 📈 Key Findings
* **Discrimination:** Logistic Regression and the Stacking Ensemble achieved identical top performance (**ROC-AUC: 0.730**, **PR-AUC: 0.609**). Stacking provided no clear advantage over the linear baseline.
* **Calibration:** Sigmoid calibration significantly improved probability reliability. Logistic Regression achieved the lowest calibrated Expected Calibration Error (**ECE: 0.0058**) and log loss (**0.5637**).
* **Explanation Stability:** Tree-based models demonstrated higher robustness; LightGBM achieved a **SHAP-rank correlation of 0.926** and **top-5 Jaccard overlap of 0.800**, outperforming Logistic Regression.

## 📂 Repository Structure
```text
├── data/                  # Placeholder for Lending Club loan dataset
├── src/
│   ├── preprocessing.py   # Winsorization, imputation, encoding, and class weights
│   ├── models.py          # Baseline, boosting, and stacking ensemble code
│   ├── calibration.py     # Platt scaling / Isotonic regression logic
│   └── explainability.py  # SHAP calculations and bootstrap stability analysis
├── notebooks/
│   └── evaluation.ipynb   # Main experimental workflow and plots
├── README.md              # Project documentation
└── requirements.txt       # Software dependencies
```

## 🛠️ Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ranamohsincodes
   cd loan-default-xai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📄 Citation
If you use this framework or reference the findings in your research, please cite:
```text
Qadir, A., Abbas, T., Mohsin, R., & Tahir, M. (2026). Explainable Ensemble Learning for Loan Default Prediction: An Evaluation of Probability Calibration and Explanation Stability. International Journal of Advanced Research, 3(6), 4474-4492. DOI: https://doi.org
```
