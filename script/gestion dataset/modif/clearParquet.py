import pandas as pd

input_file = "../../dataset/OMDS-CTD datalight_with_pos.parquet"          # chemin vers ton fichier source
output_file = "../../dataset/OMDS-CTD datalight clear.parquet"  # chemin du fichier de sortie

df = pd.read_parquet(input_file)

# Convertir la colonne 'longitude' en numérique (forcera NaN si non convertible)
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

# Supprimer les lignes où la longitude est < -69.7 et > -64
df_filtered = df[~((df["longitude"] < -69.7) & (df["longitude"] > -59) & (df["latitude"] < 47))]

# Sauvegarder le fichier filtré
df_filtered.to_parquet(output_file, index=False)

print(f"Fichier filtré enregistré : {output_file}")
