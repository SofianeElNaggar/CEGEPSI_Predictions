import pandas as pd
from pathlib import Path

CSV_PATH = Path(r"../../dataset/OMDS-CTD-meteogc-data.csv")
COLUMN = "latitude (degrees north)"
MIN_VAL = 45.5
MAX_VAL = 51.6

# Chargement du CSV
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")  # ajouter sep=";" si le séparateur n'est pas une virgule

# s'assure que la colonne est bien numérique
df[COLUMN] = pd.to_numeric(df[COLUMN], errors="coerce")

# SUPPRESSION des lignes en dehors de l’intervalle
df = df[(df[COLUMN] >= MIN_VAL) & (df[COLUMN] <= MAX_VAL)]

# Sauvegarde (écrase le CSV)
df.to_csv(CSV_PATH, index=False, encoding="utf-8")
print("Lignes en dehors de l’intervalle supprimées.")
