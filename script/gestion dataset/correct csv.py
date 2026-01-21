import pandas as pd

# Charger le CSV
df = pd.read_csv("../../dataset/meteogc/fusion_meteo.csv", dtype=str)  # dtype=str pour éviter les conversions involontaires

# Liste des colonnes
cols = df.columns.tolist()

# Trouver l'index de la colonne G
idx_G = 6

# Décaler vers la droite toutes les colonnes après G
for i in range(len(cols)-1, idx_G, -1):
    df[cols[i]] = df[cols[i-1]]

# Supprimer la colonne G (elle contient maintenant l'ancien H)
df = df.drop(df.columns[6], axis=1)

# Sauvegarder
df.to_csv("../../dataset/meteogc/meteogc dataset.csv", index=False)
