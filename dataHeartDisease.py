import ssl
import certifi
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Permet de contourner les problèmes de certificat SSL lors du téléchargement des données
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# ----------------------------------------------------------------
# ---------- Fonctions pour la manipulation des données ----------
# ----------------------------------------------------------------

def get_num_cols(df, cat_cols):
    # variable numérique : age, trestbps, chol, thalach, oldpeak
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [col for col in num_cols if col not in cat_cols] # mise à jour de num_cols pour exclure les variables catégorielles ajoutées
    
    return num_cols

def get_cat_cols(df):
    # variable catégorielle : sex, cp, fbs, restecg, exang, slope, ca, thal 
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = list(cat_cols) + ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'] # ajout des variables catégorielles 
    return cat_cols 

def get_df():
    # recuper toute les données du dataset
    return df


def show_info(df):
    # Affichage des informations sur le dataset
    # -----------------------------------------
    # regarde la forme du dataset, les colonnes, les types de données et les valeurs manquantes
    print("Forme du dataset :")
    print(df.shape)
    print("------------------------------")
    # affiche les premières lignes du dataset
    print("Aperçu du dataset :")
    print(df.head(10))
    print("------------------------------")
    # affiche les noms des colonnes et les types de données
    print("Noms des colonnes :")
    print(df.columns)
    print("------------------------------")
    # affiche les types de données de chaque colonne
    print("Types de données par colonne :")
    print(df.dtypes)
    print("------------------------------")
    # affiche le nombre de valeurs manquantes pour chaque colonne
    print("Valeurs manquantes par colonne :")
    print(df.isnull().sum())
    print("------------------------------")
    # affiche les statistiques descriptives du dataset, y compris la moyenne, l'écart type, les valeurs minimales et maximales, etc.
    print("Statistiques descriptives :")
    print(df.describe())

# ----------------------------------------------------------------
# ----------- Chargement et préparation des données --------------
# ----------------------------------------------------------------

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1) # df final avec les features et la variable cible

# ajout num_target pour indiquer la présence ou l'absence de maladie cardiaque (binaire)
df['num_target'] = (df['num'] > 0).astype(int) 

#show_info(df)