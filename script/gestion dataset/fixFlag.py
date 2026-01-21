import os
import pandas as pd

root_folder = r"../../dataset/meteogc"

for subdir, dirs, files in os.walk(root_folder):
    for file in files:
        if file.lower().endswith(".csv"):
            file_path = os.path.join(subdir, file)
            print(f"Traitement : {file_path}")

            # Lire seulement les colonnes via pandas
            df = pd.read_csv(file_path, header=2, nrows=0)
            columns = df.columns.tolist()

            # Renommer colonnes Flag sans toucher aux autres
            new_columns = []
            flag_count = 1
            for col in columns:
                if col == "Flag":
                    new_columns.append(f"Flag{flag_count}")
                    flag_count += 1
                else:
                    new_columns.append(col)

            # Lire toutes les lignes du fichier
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Réécrire UNIQUEMENT la ligne 3
            lines[2] = ",".join(new_columns) + "\n"

            # Réécrire le fichier complet (inchangé sauf ligne 3)
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            print("✔ Renommage terminé\n")
