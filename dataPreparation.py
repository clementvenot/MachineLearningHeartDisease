from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataHeartDisease import get_cat_cols, get_df, get_num_cols


def prepare_data_for_ml(test_size=0.2, random_state=42):
    """
    Pipeline de preparation des donnees:
    1) Separation train/test
    2) Conversion numerique puis imputation des valeurs manquantes (most_frequent)
    3) Reequilibrage des classes avec SMOTE (train uniquement)
    4) Re-identification des colonnes numeriques et categorielles
    5) One-hot encoding sur les donnees train reequilibrees
    6) Alignement des colonnes train/test
    7) Normalisation des colonnes numeriques

    Objectif general:
    - Eviter les fuites de donnees (les transformations sont apprises sur train)
    - Donner un format 100% numerique et homogene pour les modeles ML
    - Corriger le desequilibre de classes uniquement sur train
    """
    df = get_df().copy()

    # Cree une cible binaire a partir de la colonne multiclasses `num` si necessaire.
    # 1 = presence de maladie, 0 = absence.
    if "num_target" not in df.columns:
        df["num_target"] = (df["num"] > 0).astype(int)

    X = df.drop(columns=["num", "num_target"], errors="ignore")
    y = df["num_target"]

    # 1) Separer les donnees en ensembles train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 2) SMOTE ne fonctionne qu'avec des donnees numeriques.
    # On convertit d'abord; les valeurs non convertibles deviennent NaN.
    X_train_numeric = X_train.apply(pd.to_numeric, errors="coerce")
    X_test_numeric = X_test.apply(pd.to_numeric, errors="coerce")

    # Puis on impute en une seule passe (train -> test) avec `most_frequent`.
    # Cette etape couvre a la fois les NaN d'origine et ceux crees par conversion.
    imputer = SimpleImputer(strategy="most_frequent")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_numeric),
        columns=X_train_numeric.columns,
        index=X_train_numeric.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test_numeric),
        columns=X_test_numeric.columns,
        index=X_test_numeric.index,
    )

    # 3) Reequilibrage avec SMOTE sur train uniquement.
    # On genere des exemples synthetiques de la classe minoritaire sans toucher
    # au test, afin de conserver une evaluation representative du monde reel.
    smote = SMOTE(random_state=random_state) 
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_imputed, y_train) 
    X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train_imputed.columns) 
    y_train_balanced = pd.Series(y_train_balanced, name="num_target") 

    # 4) Re-identifier les attributs numeriques et categoriels
    cat_cols = get_cat_cols(X_train_balanced)
    num_cols = get_num_cols(X_train_balanced, cat_cols)

    # 5) Encodage one-hot: transforme les categories en colonnes binaires
    # exploitables par la plupart des algorithmes de ML.
    X_train_encoded = pd.get_dummies(X_train_balanced, columns=cat_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_imputed, columns=cat_cols, drop_first=True)

    # 6) Alignement train/test:
    # - garantit le meme ordre et le meme nombre de features
    # - ajoute les colonnes absentes dans test avec 0
    # Ceci evite les erreurs de dimension au moment du fit/predict.
    X_train_aligned, X_test_aligned = X_train_encoded.align(
        X_test_encoded,
        join="left",
        axis=1,
        fill_value=0,
    )
    # 7) Normalisation des colonnes numeriques uniquement.
    # Le scaler est ajuste sur train puis applique a test pour eviter la fuite.
    num_cols_to_scale = [col for col in num_cols if col in X_train_aligned.columns]
    scaler = StandardScaler()
    X_train_aligned[num_cols_to_scale] = scaler.fit_transform(X_train_aligned[num_cols_to_scale])
    X_test_aligned[num_cols_to_scale] = scaler.transform(X_test_aligned[num_cols_to_scale])

    return {
        "X_train": X_train_aligned,
        "X_test": X_test_aligned,
        "y_train": y_train_balanced,
        "y_test": y_test.reset_index(drop=True),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "imputer": imputer,
        "scaler": scaler,
    }


if __name__ == "__main__":
    data = prepare_data_for_ml()
    print("X_train shape:", data["X_train"].shape)
    print("X_test shape:", data["X_test"].shape)
    print("y_train distribution after SMOTE:")
    print(data["y_train"].value_counts())
