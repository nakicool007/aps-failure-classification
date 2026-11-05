# Homework 6: APS Failure Classification and Statistical Learning

Homework 6 by Nakul Kumbria (`nakicool007`). The [completed notebook](notebook/Kumbria_Nakul_HW6.ipynb) contains all five homework questions: APS failure classification, regularization, ensemble predictions, and a maximal-margin classifier example. Written answers, calculations, and executed plots are included.

## Files

```text
.
├── README.md
├── .gitignore
├── data
│   └── aps_failure_data
│       ├── aps_failure_description.txt
│       ├── aps_failure_training_set.csv
│       └── aps_failure_test_set.csv
├── notebook
│   └── Kumbria_Nakul_HW6.ipynb
└── requirements.txt

3 directories, 7 files
```

## Setup and usage

The notebook was executed with **Python 3.14**. From the project root, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open `notebook/Kumbria_Nakul_HW6.ipynb` in VS Code with the Python and Jupyter extensions installed. Choose **Select Kernel → Python Environments** and select `.venv\Scripts\python.exe`, then choose **Run All**. If the environment is not listed, use **Enter interpreter path** to select that executable.

Saved outputs are included in the notebook. Its data loader works when the kernel starts in either the project root or the `notebook/` directory.

Questions 2-4 contain written solutions. Question 5 uses a small dataset defined directly in its code cell. Run the setup cell first if running Question 5 separately. Question 1 includes five-fold cross-validation and SMOTE on the full APS training set.

## Dataset

Source: [UCI Machine Learning Repository — APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks).

Both CSV files and the dataset description are in `data/aps_failure_data/`. To restore them, download the dataset from UCI, extract the archive, and place the training and test CSV files in that directory with the names shown above. The loader detects the header after the license preamble and interprets `na` as missing data.

The training set contains 60,000 observations and 170 numeric features, plus the class label. There are 59,000 negatives and 1,000 positives, a 59:1 imbalance. The test set contains 16,000 observations. Positive labels indicate APS component failures; negative labels indicate failures unrelated to the APS. Preserve the dataset's supplied copyright and license notices when redistributing its files.

## Notebook contents

### Question 1: APS failure classification

The notebook covers assignment parts **1(a)**, **1(b)(i–v)**, **1(c)**, **1(d)**, **1(e)**, and **1(f)**:

- Load and validate the training and test data.
- Discuss missing-value methods and apply median imputation.
- Calculate the coefficient of variation for all 170 features using sample standard deviations.
- Plot the full Pearson correlation matrix.
- Visualize the 13 highest-CV features with a pairwise scatterplot matrix and class-wise boxplots.
- Report class counts and explain the imbalance and limits of visual feature assessment.
- Fit a 300-tree random forest without class weighting or resampling, report training and test confusion matrices, ROC curves, AUC, and misclassification rates, and compare out-of-bag error with test error.
- Research imbalance handling and repeat the forest evaluation with `class_weight="balanced_subsample"`, comparing directly with the unweighted forest.
- Select XGBoost's L1 regularization strength using five-fold cross-validation and compare its CV error with the test error.
- Repeat the XGBoost experiment with SMOTE inside every fitting fold, reporting confusion matrices, ROC curves, AUC, and misclassification rates for the original training and test observations.

### Question 2: Lasso and the bias-variance trade-off

Explains how increasing the lasso constraint bound affects training RSS, test RSS, variance, squared bias, and irreducible error, with a choice and justification for every part.

### Question 3: Ridge and lasso with identical predictors

Derives both optimization problems, proves that ridge gives equal coefficients, and describes the lasso solution set, including the unique zero-solution exception.

### Question 4: Combining bootstrap-tree predictions

Shows that majority voting predicts Red (6 votes to 4), whereas averaging the estimated Red probabilities gives 0.45 and predicts Green. Explains why the methods disagree.

### Question 5: Maximal-margin classification

Answers parts (a)-(h) with four labeled plots. Derives the separator `X2 = X1 - 0.5`, the margin `1 / (2 sqrt(2))`, and support-vector observations 2, 3, 5, and 6. Also explains why moving observation 7 slightly does not change the separator, gives a nonoptimal separating line, and adds a Blue point at `(3, 3)` to destroy linear separability.

## Modeling and reproducibility

Imputation is fitted on training data only and applied to both splits. CVs, correlations, and boxplots use all training observations. The scatterplots use a reproducible sample of 3,000 negatives and 1,000 positives, with `log(1 + value)` axes for readability. This sample is only for visualization; all 170 features remain available for subsequent modeling.

All APS models use all 170 features, fixed seeds, and no test-based tuning. Part (c) uses the original class proportions without compensation; part (d) uses per-bootstrap class weights. SMOTE is applied only in part (f). For XGBoost, imputation, log transformation, scaling, and any resampling stay inside each fitting fold. Final test observations are never resampled.

Following the instructor clarification supplied for 1(e), standard XGBoost with cross-validated L1 regularization is sufficient; a separate logistic regression model tree is not required. Parts (e)–(f) use XGBoost trees with binary logistic loss and cross-validated `reg_alpha`, consistent with that clarification.

The requirements pin the versions used for execution. The `.gitignore` excludes the local virtual environment, caches, notebook checkpoints, logs, and local configuration while retaining the notebook, its saved outputs, and the supplied CSV data.

## References

- [scikit-learn: Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)
- [scikit-learn: SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [imbalanced-learn: common pitfalls and resampling leakage](https://imbalanced-learn.org/stable/common_pitfalls.html)
