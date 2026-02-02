import pandas as pd
import os

root_folder = "../../dataset/meteogc"
all_data = []

for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)

    if not os.path.isdir(subfolder_path):
        continue

    for file in os.listdir(subfolder_path):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(subfolder_path, file)
        print(file_path)

        # --- Lire uniquement les deux premières lignes pour extraire les métadonnées ---
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [next(f) for _ in range(3)]   # 3 premières lignes

        # Les lignes sont séparées par des virgules
        row2 = lines[1].strip().split(",")

        station_name = row2[1]
        province     = row2[2]
        station_id   = row2[4]
        latitude     = row2[5]
        longitude    = row2[6]
        utc_offset   = row2[7]

        # --- Lecture du CSV réel : à partir de la ligne 3 ---
        df = pd.read_csv(file_path, skiprows=2, header=0, low_memory=False)

        # --- Ajouter les métadonnées devant ---
        df.insert(0, "Station Name", station_name)
        df.insert(1, "Station ID", station_id)
        df.insert(2, "Province", province)
        df.insert(3, "Latitude", latitude)
        df.insert(4, "Longitude", longitude)
        df.insert(5, "UTC offset", utc_offset)

        all_data.append(df)

# --- Fusionner tous les fichiers ---
final_df = pd.concat(all_data, ignore_index=True)

# --- Export final ---
output = "../../dataset/meteogc/fusion_meteo.csv"
final_df.to_csv(output, index=False)

print("Fichier final créé :", output)
