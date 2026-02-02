import csv
import tempfile
import os

# Chemin vers le fichier CSV
csv_path = "../../dataset/OMDS-CTD-meteogc-data.csv"

# Numéro de la ligne à supprimer (1 = première ligne)
line_to_delete = 2

# Créer un fichier temporaire
with tempfile.NamedTemporaryFile('w', delete=False, newline='', encoding='utf-8') as tmpfile:
    with open(csv_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        writer = csv.writer(tmpfile)

        for i, row in enumerate(reader, start=1):
            if i == line_to_delete:
                continue  # sauter la ligne à supprimer
            writer.writerow(row)

# Remplacer le fichier original par le fichier temporaire
os.replace(tmpfile.name, csv_path)

print(f"La ligne {line_to_delete} a été supprimée du CSV.")
