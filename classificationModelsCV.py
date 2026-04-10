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
    # On reprend la base du script de classification: donnees preparees par le pipeline.
    data = prepare_data_for_ml(test_size=test_size, random_state=random_state)
    x_train = data["X_train"]
    y_train = data["y_train"]

    # Meme liste de modeles que dans le script principal.
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

    # Validation croisee stratifiee:
    # - Le train est decoupe en 5 sous-ensembles (folds)
    # - A chaque iteration: 4 folds pour entrainer, 1 fold pour valider
    # - La stratification conserve la proportion 0/1 dans chaque fold
    # - Le shuffle + random_state assurent un decoupage melange mais reproductible
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for model_name, model in models.items():
        # cross_val_score entraine/valide automatiquement sur chaque fold
        # et retourne un score d'accuracy par fold.
        fold_scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=None,
        )

        # On resume les resultats de CV par modele:
        # - moyenne: performance globale attendue
        # - ecart-type: stabilite du modele selon les folds
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
