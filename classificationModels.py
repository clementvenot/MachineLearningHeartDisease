from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from dataPreparation import prepare_data_for_ml


def evaluate_classification_models(test_size=0.2, random_state=42):
    """Entraine et compare plusieurs modeles de classification sur les donnees de maladie cardiaque."""
    # On recupere les donnees deja preparees par le pipeline.
    # Cela garantit que tous les modeles recoivent exactement les memes features.
    data = prepare_data_for_ml(test_size=test_size, random_state=random_state)
    x_train = data["X_train"]
    x_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Dossier de sortie pour les matrices de confusion.
    output_dir = Path("results/confusion_matrices")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Liste des modeles testes dans l'etude.
    # Le but est de comparer des approches differentes:
    # - distance (kNN)
    # - lineaire (regression logistique)
    # - arbre de decision
    # - methode a noyau (SVM)
    # - probabiliste (Naive Bayes)
    # - reseau de neurones (MLP)
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
    

    # Cette liste va stocker les resultats de chaque modele.
    rows = []
    for model_name, model in models.items():
        # Entrainement du modele sur le jeu d'apprentissage.
        model.fit(x_train, y_train)

        # Prediction sur le jeu de test pour evaluer la capacite de generalisation.
        y_pred = model.predict(x_test)

        # Trace et sauvegarde la matrice de confusion pour ce modele.
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        display.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Matrice de confusion - {model_name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=150)
        plt.close(fig)

        row = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
        }

        rows.append(row)

    # Conversion des resultats en DataFrame pour obtenir un tableau lisible.
    # Le tri par accuracy permet de classer les modeles du meilleur au moins bon.
    results = pd.DataFrame(rows).sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    return results


if __name__ == "__main__":
    # Execution du script en ligne de commande: on calcule les performances des modeles.
    results_df = evaluate_classification_models()

    # Parametrage d'affichage pour voir toutes les colonnes dans le terminal.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    print("Comparaison des modeles de classification:")
    print(results_df.round(4).to_string(index=False))
    print("Matrices de confusion enregistrees dans: results/confusion_matrices")
