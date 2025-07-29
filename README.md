# Discrete Time Survival Analysis
## __Table of Contents__ ##

<ul>

[1. About the project](#about-the-project)

[2. About the dataset](#about-the-dataset)

[3. Tools and libraries](#tools-and-libraries)

[4. Specifications and Setup](#spec-and-setup)

[5. Results](#results)

</ul>

<hr>

## __About the project__ ##

This repository adapts the original [autoSurv](https://github.com/ksuresh17/autoSurv) codebase — originally implemented in R — into Python. The goal is to provide a modular and flexible framework for benchmarking and applying classical machine learning and survival models for survival analysi tasks.

This adaptation is tailored for research use, model benchmarking, and reproducibility experiments in survival analysis, particularly under settings involving class imbalance or the need for discrete-time modeling techniques.

### 🔍 Models Available

The following models are implemented:

- **Cox Proportional Hazards (CoxPH)** 
- **Neural Network** 
- **Logistic Regression**
- **XGBoost** 
- **Support Vector Machine (SVM)** 
- **Random Forest** 
- **Random Survival Forest (RSF)** 
- **Neural Network with Focal Loss** 

This project supports both continuous and discretized survival formulations.

---

## __About the dataset__ ##

This repository is compatible with any survival dataset that meets the following criteria:

- **Normalized**: Continuous features are scaled appropriately.
- **One-hot encoded**: Categorical variables are transformed into binary columns.
- **Right Censored**

For demonstration and testing purposes, a normalized and encoded version of the **PBC dataset** (Primary Biliary Cirrhosis) is included.

📎 Original dataset source: [PBC - survival package](https://rdrr.io/cran/survival/man/pbc.html)

---

## __Tools and libraries__ ##

This project was developed in Python, for machine learning and survival analysis tasks.

The main libraries used are:

* __Pandas and NumPy__: for data manipulation, cleaning, and numerical operations.

* __Scikit-learn__: for traditional ML models such as logistic regression, SVM, random forest, and Neural Nets.

* __XGBoost__: for gradient boosting classification.

* __Lifelines__: for Cox Proportional Hazards modeling and concordance metrics.

* __Scikit-survival__: for random survival forests and concordance metrics.

* __Scikit-optimize (skopt)__: for hyperparameter tuning using Bayesian optimization.

* __TensorFlow__: for implementing Neural Nets when using focal loss.

* __Focal-loss__: for applying binary focal loss during training.

---

## __Specifications and Setup__ ##

Each model is trained and evaluated using **5-fold cross-validation**, and the final results are **averaged across the folds** and returned in a **DataFrame format** for easy comparison and analysis.

### ⚙️ How to Run the Code

1. Install all necessary packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the main pipeline:
    ```
    python3 main.py
    ```

3. If you'd like to test on a different dataset:
- Go to main.py

- Replace the dataset loading section with your preprocessed data (ensure it follows the format described above).

---
## __Results__ ##

The following table summarizes the performance of different models on the PBC dataset using 5-fold cross-validation. The reported metrics are averaged across the folds:

| Model        | C-index   | Mean Brier | Mean PR-AUC | Mean ROC-AUC |
|--------------|-----------|------------|--------------|---------------|
| CoxPH_Train  | 0.849753  | 0.206721   | 0.786591     | 0.759598      |
| CoxPH_Val    | 0.827699  | 0.217404   | 0.770339     | 0.747275      |
| LogReg_Train | —         | 0.208020   | 0.825718     | 0.861255      |
| LogReg_Val   | —         | 0.210058   | 0.812375     | 0.848784      |
| NNet_Train   | —         | 0.251181   | 0.885767     | 0.907350      |

Note:
- “—” indicates the metric is not applicable for that model.

---

## __Citation__ ##

- Suresh, K., Severn, C. & Ghosh, D. Survival prediction models: an introduction to discrete-time modeling. BMC Med Res Methodol 22, 207 (2022). https://doi.org/10.1186/s12874-022-01679-6

---