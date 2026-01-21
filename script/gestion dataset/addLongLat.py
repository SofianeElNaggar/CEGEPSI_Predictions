import pandas as pd

# --- Coordonnées de remplacement : Sept-Îles, QC, Canada ---
DEFAULT_LAT = 50.2011
DEFAULT_LON = -66.3821

# --- Charger le CSV ---
df = pd.read_csv("../../dataset/Tided Updated OMDS-CTD data.csv")  # Remplace par ton fichier

# --- Remplacer les valeurs manquantes ---
df['latitude'] = df['latitude'].fillna(DEFAULT_LAT)
df['longitude'] = df['longitude'].fillna(DEFAULT_LON)

# --- Sauvegarder le résultat ---
df.to_csv("../../dataset/Tide OMDS-CTD data.csv", index=False)

print("Traitement terminé : output.csv créé.")
