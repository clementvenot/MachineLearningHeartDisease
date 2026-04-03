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
    2) Imputation des valeurs manquantes (most_frequent)
    3) Reequilibrage des classes avec SMOTE (train uniquement)
    4) Re-identification des colonnes numeriques et categorielles
    5) One-hot encoding sur les donnees train reequilibrees
    6) Alignement des colonnes train/test
    7) Normalisation des colonnes numeriques
    """
    df = get_df().copy()

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

    # 2) Imputer les valeurs manquantes avec la strategie most_frequent
    imputer = SimpleImputer(strategy="most_frequent")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # SMOTE exige des valeurs numeriques
    X_train_imputed = X_train_imputed.apply(pd.to_numeric, errors="coerce")
    X_test_imputed = X_test_imputed.apply(pd.to_numeric, errors="coerce")
    X_train_imputed = X_train_imputed.fillna(X_train_imputed.mode().iloc[0])
    X_test_imputed = X_test_imputed.fillna(X_train_imputed.mode().iloc[0])

    # 3) Reequilibrer les classes avec SMOTE (uniquement sur le train)
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_imputed, y_train)
    X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train_imputed.columns)
    y_train_balanced = pd.Series(y_train_balanced, name="num_target")

    # 4) Re-identifier les attributs numeriques et categoriels
    cat_cols = get_cat_cols(X_train_balanced)
    num_cols = get_num_cols(X_train_balanced, cat_cols)

    # 5) Appliquer le one-hot encoding sur les donnees reequilibrees
    X_train_encoded = pd.get_dummies(X_train_balanced, columns=cat_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_imputed, columns=cat_cols, drop_first=True)

    # 6) Aligner les colonnes entre train et test
    X_train_aligned, X_test_aligned = X_train_encoded.align(
        X_test_encoded,
        join="left",
        axis=1,
        fill_value=0,
    )

    # 7) Normaliser les attributs numeriques
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
