import matplotlib.pyplot as plt
from dataHeartDisease import  get_df, get_num_cols, get_cat_cols
from dataHotEncoding import get_encoded_df

# ---------------------------------------------------------------
# ----------------- Fonction de visualisation -------------------
# ---------------------------------------------------------------

def show_num_histograms(df, num_cols):
    # Affichage des histogrammes pour les variables numériques
    for col in num_cols:
        plt.figure()
        df[col].hist()
        plt.title(f"Histogramme de {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.show()

def show_cat_barplots(df, cat_cols):
    # Affichage des barplots pour les variables catégorielles
    for col in cat_cols:
        plt.figure()
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Distribution de {col}")
        plt.xlabel(col)
        plt.ylabel("Nombre")
        plt.show()

def show_target_distribution(df):
    # Analyse de la variable cible
    plt.figure()
    df['num'].value_counts().plot(kind='bar')
    plt.title("Distribution de la variable cible (num)")
    plt.xlabel("Classe")
    plt.ylabel("Nombre")
    plt.show()

def show_binary_target_distribution(df):
    # Analyse de la variable cible binaire
    plt.figure()
    df['num_target'].value_counts().plot(kind='bar')
    plt.title("Distribution de la cible binaire")
    plt.xlabel("0 = pas malade | 1 = malade")
    plt.ylabel("Nombre")
    plt.show()

# variables numériques
df = get_df()
cat_cols = get_cat_cols(df)
num_cols = get_num_cols(df, cat_cols)

# Affichage des histogrammes pour les variables numériques
show_num_histograms(df, num_cols)
# Affichage des barplots pour les variables catégorielles
show_cat_barplots(df, cat_cols)
# Analyse de la variable cible
show_target_distribution(df)
# Analyse de la variable cible binaire
show_binary_target_distribution(df)