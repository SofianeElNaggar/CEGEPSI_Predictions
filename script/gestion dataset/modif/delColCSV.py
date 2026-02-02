import pandas as pd

csv_path = r"../../dataset/OMDS-CTD-meteogc-data.csv"
col_to_remove = "station"
output_path = r"../../dataset/OMDS-CTD-meteogc-data.csv"

# --- Essais d'encodages ---
encodings_to_try = ["utf-8", "latin1", "cp1252"]

for enc in encodings_to_try:
    try:
        df = pd.read_csv(csv_path, encoding=enc)
        print(f"Fichier lu correctement avec l'encodage : {enc}")
        break
    except UnicodeDecodeError:
        print(f"Échec avec encodage : {enc}")
else:
    raise ValueError("Impossible de lire le fichier avec les encodages testés.")

# --- Suppression de la colonne ---
if col_to_remove in df.columns:
    df = df.drop(columns=[col_to_remove])
    print(f"Colonne '{col_to_remove}' supprimée.")
else:
    print(f"La colonne '{col_to_remove}' n'existe pas.")

# --- Sauvegarde ---
df.to_csv(output_path, index=False, encoding=enc)
print(f"Fichier modifié enregistré sous : {output_path}")
