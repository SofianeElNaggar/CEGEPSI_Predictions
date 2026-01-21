import pandas as pd
import os

# Chemin du fichier CSV d'origine
csv_path = "../../dataset/OMDS-CTD data.csv"
# Chemin du fichier Parquet à créer
parquet_path = os.path.splitext(csv_path)[0] + "light_with_pos.parquet"

# Liste des colonnes à conserver
columns_to_keep = [
    "time",
    "depth",
    "PRESPR01",
    "PRESPR01_QC",
    "TEMPS901",
    "TEMPS901_QC",
    "CPHLPR01",
    "TURBPR01",
    "PHXXPR01",
    "PHXXPR01_QC",
    "PSALST01",
    "PSALST01_QC",
    "SIGTEQST",
    "SIGTEQST_QC",
    "DOXYZZ01",
    "DOXYZZ01_QC",
    "latitude",
    "longitude"
]

def csv_to_parquet_filtered(csv_path, parquet_path, columns):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    print(f"Lecture du fichier CSV : {csv_path}")
    # Lecture uniquement des colonnes nécessaires
    df = pd.read_csv(csv_path, usecols=columns, low_memory=False)

    print(f"Conversion en Parquet : {parquet_path}")
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    print("Conversion terminée avec succès ✅")
    print(f"Fichier créé : {parquet_path}")
    print(f"Colonnes conservées : {list(df.columns)}")

# Exécution directe
if __name__ == "__main__":
    csv_to_parquet_filtered(csv_path, parquet_path, columns_to_keep)
