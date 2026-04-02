import matplotlib.pyplot as plt
from dataHeartDisease import  get_df
df = get_df()

# Variables numériques
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Variables catégorielles
cat_cols = df.select_dtypes(include=['object', 'category']).columns

cat_cols = list(cat_cols) + ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'] # ajout des variables catégorielles 
num_cols = [col for col in num_cols if col not in cat_cols] # mise à jour de num_cols pour exclure les variables catégorielles ajoutées

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