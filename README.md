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

For demonstration and testing purposes, a normalized and encoded version of the **PBC dataset** (Primary Biliary Cirrhosis) is included in the data folder.

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

| Model            | C-index   | Mean Brier | Mean PR-AUC | Mean ROC-AUC |
|------------------|-----------|------------|-------------|--------------|
| CoxPH_Train      | 0.8498    | 0.2067     | 0.7866      | 0.7596       |
| CoxPH_Val        | 0.8277    | 0.2174     | 0.7703      | 0.7473       |
| LogReg_Train     | —         | 0.2080     | 0.8257      | 0.8613       |
| LogReg_Val       | —         | 0.2101     | 0.8124      | 0.8488       |
| NNet_Train       | —         | 0.2512     | 0.8858      | 0.9074       |
| NNet_Val         | —         | 0.2599     | 0.8282      | 0.8667       |
| NNetFocal_Train  | —         | 0.2344     | 0.8060      | 0.8591       |
| NNetFocal_Val    | —         | 0.2363     | 0.8013      | 0.8509       |
| RF_Train         | —         | 0.2626     | 0.8980      | 0.9183       |
| RF_Val           | —         | 0.2768     | 0.8231      | 0.8491       |
| RSF_Train        | 0.8733    | 0.1899     | 0.8202      | 0.8047       |
| RSF_Val          | 0.8305    | 0.2063     | 0.7726      | 0.7594       |
| SVM_Train        | —         | 0.2954     | 0.8355      | 0.8821       |
| SVM_Val          | —         | 0.3047     | 0.7126      | 0.7505       |
| XGBoost_Train    | —         | 0.3344     | 1.0000      | 1.0000       |
| XGBoost_Val      | —         | 0.3613     | 0.8126      | 0.8456       |

Note:
- “—” indicates the metric is not applicable for that model.

---

## __Citation__ ##

- Suresh, K., Severn, C. & Ghosh, D. Survival prediction models: an introduction to discrete-time modeling. BMC Med Res Methodol 22, 207 (2022). https://doi.org/10.1186/s12874-022-01679-6

---