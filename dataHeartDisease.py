import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

from ucimlrepo import fetch_ucirepo
import pandas as pd

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)
df['num_target'] = (df['num'] > 0).astype(int) # ajout num_target pour indiquer la présence ou l'absence de maladie cardiaque (binaire)


# Affichage des informations sur le dataset

# regarde la forme du dataset, les colonnes, les types de données et les valeurs manquantes
print("Forme du dataset :")
print(df.shape)

# affiche les premières lignes du dataset
print("------------------------------")
print("Aperçu du dataset :")
print(df.head(10))

# affiche les noms des colonnes et les types de données
print("------------------------------")
print("Noms des colonnes :")
print(df.columns)

# affiche les types de données de chaque colonne
print("------------------------------")
print("Types de données par colonne :")
print(df.dtypes)

# affiche le nombre de valeurs manquantes pour chaque colonne
print("------------------------------")
print("Valeurs manquantes par colonne :")
print(df.isnull().sum())

# affiche les statistiques descriptives du dataset, y compris la moyenne, l'écart type, les valeurs minimales et maximales, etc.
print("------------------------------")
print("Statistiques descriptives :")
print(df.describe())