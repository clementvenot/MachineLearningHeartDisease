
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

<img width="451" height="129" alt="image" src="https://github.com/user-attachments/assets/d1e0767a-f0ee-4367-9528-8daf446f6e8f" />

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

<img width="1314" height="433" alt="Screenshot 2026-04-10 at 15 14 01" src="https://github.com/user-attachments/assets/a538630c-7a83-479b-b75e-d5bea0b1be3a" />
<img width="906" height="399" alt="Screenshot 2026-04-10 at 15 14 47" src="https://github.com/user-attachments/assets/52ff925a-acec-4666-bb73-fad924c1f2bd" />

## Tested Models

In [classificationModels.py](classificationModels.py):

- kNN
- Logistic Regression
- Decision Tree
- SVM (RBF)
- Naive Bayes
- MLPClassifier


<img width="171" height="82" alt="image" src="https://github.com/user-attachments/assets/8b992100-49b8-4d49-b03e-302374b31c81" />

## Cross-Validation

In [classificationModelsCV.py](classificationModelsCV.py): 5-fold stratified cross-validation with accuracy (mean and standard deviation per model).

<img width="222" height="81" alt="image" src="https://github.com/user-attachments/assets/ba9568a9-9252-4258-a2a1-3e8ea7517c68" />

Confusion matrices are saved in [results/confusion_matrices](results/confusion_matrices).

<img width="191" height="153" alt="image" src="https://github.com/user-attachments/assets/b4aeccc0-8ccc-48bb-b568-d38deb2b01cf" /><img width="191" height="153" alt="image" src="https://github.com/user-attachments/assets/27ee1410-4e8c-46e2-9a3e-733390296f6d" />

***

## License

There is no license; you're free to use it.

Feel free to contribute to this project by submitting issues or pull requests.

For any questions or support, please contact [Clément Venot](mailto:clement.venooot@gmail.com).
