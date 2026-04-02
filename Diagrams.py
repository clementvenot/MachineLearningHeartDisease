
import matplotlib.pyplot as plt


# Variables numériques
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Variables catégorielles
cat_cols = df.select_dtypes(include=['object', 'category']).columns

print("Numériques :", num_cols)
print("Catégorielles :", cat_cols)



for col in num_cols:
    plt.figure()
    df[col].hist()
    plt.title(f"Histogramme de {col}")
    plt.xlabel(col)
    plt.ylabel("Fréquence")
    plt.show()