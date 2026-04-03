from dataHeartDisease import get_df, get_num_cols, get_cat_cols, show_info
import pandas as pd

def get_encoded_df():
    # récupérer les colonnes catégorielles du dataset pour les encoder en utilisant pd.get_dummies 
    df = get_df()
    cat_cols = get_cat_cols(df)
    
    # Encodage des variables catégorielles en utilisant pd.get_dummies
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_encoded = df_encoded.drop(columns=['num'], errors='ignore')
    df_encoded = df_encoded.astype(int)

    return df_encoded

if __name__ == "__main__":
    # encoder
    df_encoded = get_encoded_df()
    show_info(df_encoded)