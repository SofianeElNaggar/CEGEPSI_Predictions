import os
import pandas as pd

# Dossier racine à parcourir
root_folder = r"../../dataset/meteogc/v2"

for folder_path, subfolders, files in os.walk(root_folder):
    for file in files:
        if file.lower().endswith(".csv"):
            csv_path = os.path.join(folder_path, file)
            print(f"Traitement : {csv_path}")

            # Charger le CSV
            df = pd.read_csv(csv_path)

            # Colonnes à supprimer : header contenant "Flag"
            cols_to_drop = [col for col in df.columns if "Flag" in col]

            if cols_to_drop:
                print(f" → Suppression des colonnes : {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)

                # Réécriture du fichier CSV
                df.to_csv(csv_path, index=False)
            else:
                print(" → Aucune colonne 'Flag' dans ce fichier.")
