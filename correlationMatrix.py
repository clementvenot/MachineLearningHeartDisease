from dataHotEncoding import get_encoded_df
import matplotlib.pyplot as plt

# Calcul de la matrice de corrélation
df = get_encoded_df()

corr_matrix = df.corr()

corr_target = corr_matrix['num_target'].sort_values(ascending=False)

# Affichage de la corrélation avec la variable cible
print(corr_target)

# Visualisation de la matrice de corrélation
plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, aspect='auto')
plt.colorbar()

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show() # Affiche la matrice de corrélation sous forme de carte de chaleur