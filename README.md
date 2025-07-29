# Discrete Time Survival Analysis
## __Table of Contents__ ##

<ul>

[1. About the project](#about-the-project)

[2. About the dataset](#about-the-dataset)

[3. Tools and libraries](#tools-and-libraries)

[4. Results](#results)

</ul>

<hr>

## __About the project__ ##

We adapted the R version from https://github.com/ksuresh17/autoSurv

Models available are:     

- Cox Proportional Hazards model
- Neural Network
- Logistic Regression
- XGBoost
- Support Vector Machine
- Random Forest
- Random Survival Forest
- Neural Network with Focal Loss

<hr>

## __About the dataset__ ##

Any dataset that has been normalized and one hot encoded for categorical variables with event and time-to-event labels can be used.

In this repository, we have made available the normalized and encoded PBC dataset which is (...) for testing puposes of this code.

## __Tools and libraries__ ##

This project was developed in Python, primarily using Jupyter Notebooks, for machine learning and survival analysis tasks.

The main libraries used are:

* __Pandas and NumPy__: for data manipulation, cleaning, and numerical operations.

* __Scikit-learn__: for traditional ML models such as logistic regression, SVM, random forest, and Neural Nets.

* __XGBoost__: for gradient boosting classification.

* __Lifelines__: for Cox Proportional Hazards modeling and concordance metrics.

* __Scikit-survival__: for random survival forests and concordance metrics.

* __Scikit-optimize (skopt)__: for hyperparameter tuning using Bayesian optimization.

* __TensorFlow__: for implementing Neural Nets when using focal loss.

* __Focal-loss__: for applying binary focal loss during training.

## __Results__ ##

