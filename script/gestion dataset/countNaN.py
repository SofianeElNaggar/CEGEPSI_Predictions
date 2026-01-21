import pandas as pd

def count_nans_in_csv(filepath, chunksize=100000):
    nan_counts = None
    total_rows = 0

    for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
        # Détection des NaN
        chunk_nans = chunk.isna()

        # Détection des valeurs vides ("", "   ", etc.)
        empty_mask = chunk.apply(lambda col: col.astype(str).str.strip() == "")

        # Combinaison NaN + vides
        combined = chunk_nans | empty_mask

        # Comptage pour ce chunk
        chunk_counts = combined.sum()

        # Accumulation
        if nan_counts is None:
            nan_counts = chunk_counts
        else:
            nan_counts += chunk_counts

        total_rows += len(chunk)

    print("Nombre total de lignes :", total_rows)
    print("Nombre de NaN / valeurs vides par colonne :")
    print(nan_counts)

# Utilisation
count_nans_in_csv("../../dataset/OMDS-CTD-meteogc-data.csv")
