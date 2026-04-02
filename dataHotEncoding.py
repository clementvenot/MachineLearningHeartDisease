from dataHeartDisease import get_df, get_num_cols, get_cat_cols, show_info
import pandas as pd

# récupérer les colonnes catégorielles et numériques du dataset original
df = get_df()
cat_cols = get_cat_cols(df)

def get_encoded_df():
    # Encodage des variables catégorielles en utilisant pd.get_dummies
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded

# encoder
df_encoded = get_encoded_df()

# recalcul des colonnes après encodage
cat_cols = get_cat_cols(df_encoded)
num_cols = get_num_cols(df_encoded, cat_cols)

#show_info(df_encoded)