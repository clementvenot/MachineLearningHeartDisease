import matplotlib.pyplot as plt
from dataHeartDisease import  get_df, get_num_cols, get_cat_cols
from dataHotEncoding import get_encoded_df

#df = get_df()
df = get_encoded_df() # Utilisation du dataset encodé


cat_cols = get_cat_cols(df)
num_cols = get_num_cols(df, cat_cols)

# --------------------------------------------------------------
# Affichage des histogrammes pour les variables numériques
print("Numériques :", num_cols)

for col in num_cols:
    plt.figure()
    df[col].hist()
    plt.title(f"Histogramme de {col}")
    plt.xlabel(col)
    plt.ylabel("Fréquence")
    plt.show()

# --------------------------------------------------------------
# Variables catégorielles
print("Catégorielles :", cat_cols)

for col in cat_cols:
    plt.figure()
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution de {col}")
    plt.xlabel(col)
    plt.ylabel("Nombre")
    plt.show()

# --------------------------------------------------------------
# Analyse de la variable cible
print("Variable cible (num) :")

plt.figure()
df['num'].value_counts().plot(kind='bar')
plt.title("Distribution de la variable cible (num)")
plt.xlabel("Classe")
plt.ylabel("Nombre")
plt.show()

# --------------------------------------------------------------
# Analyse de la variable cible binaire
print("Variable cible binaire (num_target) :")

plt.figure()
df['num_target'].value_counts().plot(kind='bar')
plt.title("Distribution de la cible binaire")
plt.xlabel("0 = pas malade | 1 = malade")
plt.ylabel("Nombre")
plt.show()