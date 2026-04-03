from dataHotEncoding import get_encoded_df

df = get_encoded_df()

corr_matrix = df.corr()

corr_target = corr_matrix['num_target'].sort_values(ascending=False)

print(corr_target)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, aspect='auto')
plt.colorbar()

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()