***

# Heart Disease Prediction - Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-0E7C86?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![UCI Dataset](https://img.shields.io/badge/UCI-Heart%20Disease-4CAF50?style=for-the-badge)
![License](https://img.shields.io/badge/License-None-lightgrey?style=for-the-badge)

This project predicts the presence of heart disease from clinical data (UCI Heart Disease dataset, id=45).
The target variable `num` is converted to a binary target:

- `0`: no disease
- `1`: disease present (`num > 0`)

***

## Workflow

The pipeline in [dataPreparation.py](dataPreparation.py) follows these steps:

1. Stratified train/test split
2. Missing value imputation (`most_frequent`)
3. Class rebalancing with SMOTE (train set only)
4. One-hot encoding of categorical variables
5. Train/test column alignment
6. Numerical feature standardization

## Exploration and Correlations

- Initial analysis: [dataHeartDisease.py](dataHeartDisease.py)
- Visualizations: [Diagrams.py](Diagrams.py)
- Correlation matrix: [correlationMatrix.py](correlationMatrix.py)

[Histograms of numerical variables]

[Bar charts of categorical variables]

[Target variable distribution]

[Correlation matrix heatmap]

## Tested Models

In [classificationModels.py](classificationModels.py):

- kNN
- Logistic Regression
- Decision Tree
- SVM (RBF)
- Naive Bayes
- MLPClassifier

Confusion matrices are saved in [results/confusion_matrices](results/confusion_matrices).

[Test set performance summary]

[Best model confusion matrix]

## Cross-Validation

In [classificationModelsCV.py](classificationModelsCV.py): 5-fold stratified cross-validation with accuracy (mean and standard deviation per model).

[Cross-validation chart]

***

## License

There is no license; you're free to use it.

---

Feel free to contribute to this project by submitting issues or pull requests.

For any questions or support, please contact [Clément Venot](mailto:clement.venooot@gmail.com).

---