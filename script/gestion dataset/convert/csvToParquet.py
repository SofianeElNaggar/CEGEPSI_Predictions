import pandas as pd
import os

# Nom du fichier CSV à convertir
csv_path = "../../dataset/OMDS-CTD-meteogc-data2.csv"

parquet_path = os.path.splitext(csv_path)[0] + ".parquet"


def csv_to_parquet(csv_path, parquet_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    print(f"Lecture du fichier CSV : {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(df.shape)
    print(df.head())

    print(f"Conversion en Parquet : {parquet_path}")
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    print("Conversion terminée avec succès !")
    print(f"Fichier créé : {parquet_path}")


# Exécution directe
if __name__ == "__main__":
    csv_to_parquet(csv_path, parquet_path)
