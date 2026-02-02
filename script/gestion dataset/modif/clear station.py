import pandas as pd

# Chemin vers ton fichier CSV
csv_path = "../../dataset/meteogc/Station Inventory EN.csv"

# Valeurs autorisées dans la colonne B
valid_values = {"PRINCE EDWARD ISLAND", "NOVA SCOTIA", "QUEBEC","NEWFOUNDLAND", "NEW BRUNSWICK"}

# Charger le CSV : header = ligne 4 → index 3
df = pd.read_csv(csv_path, header=3)

# Filtrer :
# - Colonne B (index 1) doit être dans valid_values
# - Colonne Q (index 16) doit être >= 2024
df_filtered = df[
    df.iloc[:, 1].isin(valid_values) &
    (df.iloc[:, 16] >= 2024) &
    (df.iloc[:, 15] <= 2005)
]

# Réécrire le fichier : l'en-tête est automatiquement conservé
df_filtered.to_csv(csv_path, index=False, encoding="utf-8")

