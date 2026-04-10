import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dataPreparation import prepare_data_for_ml


def evaluate_models_with_cv(n_splits=5, random_state=42, test_size=0.2):
    """
    Compare les modeles avec une validation croisee stratifiee.

    - n_splits=5: 5 folds
    - stratification: conserve la proportion des classes dans chaque fold
    - metrique: accuracy

    Returns:
        pd.DataFrame: score moyen et ecart-type par modele
    """
    # Donnees preparees par le pipeline principal.
    data = prepare_data_for_ml(test_size=test_size, random_state=random_state)
    x_train = data["X_train"]
    y_train = data["y_train"]

    # Modeles compares.
    models = {
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "NaiveBayes": GaussianNB(),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=random_state,
        ),
    }

    # Validation croisee stratifiee en 5 folds.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for model_name, model in models.items():
        # Accuracy sur chaque fold.
        fold_scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=None,
        )

        # Resume: moyenne et stabilite.
        rows.append(
            {
                "model": model_name,
                "cv_accuracy_mean": fold_scores.mean(),
                "cv_accuracy_std": fold_scores.std(),
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(by="cv_accuracy_mean", ascending=False)
        .reset_index(drop=True)
    )
    return results


if __name__ == "__main__":
    results_df = evaluate_models_with_cv()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print("Validation croisee stratifiee (5 folds) - comparaison des modeles:")
    print(results_df.round(4).to_string(index=False))
