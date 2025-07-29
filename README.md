# Discrete Time Survival Analysis
## __Table of Contents__ ##

<ul>

[1. About the project](#about-the-project)

[2. About the dataset](#about-the-dataset)

[3. Tools and libraries](#tools-and-libraries)

[4. Setup](#setup)

[5. Results](#results)

</ul>

<hr>

## __About the project__ ##

This repository adapts the original [autoSurv](https://github.com/ksuresh17/autoSurv) codebase — originally implemented in R — into Python. The goal is to provide a modular and flexible framework for benchmarking and applying classical and modern machine learning models for survival analysis.

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
- **Right Censored**:

For demonstration and testing purposes, a normalized and encoded version of the **PBC dataset** (Primary Biliary Cirrhosis) is included.

📎 Original dataset source: [PBC - survival package](https://rdrr.io/cran/survival/man/pbc.html)

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

## __Setup__ ##



## __Results__ ##

