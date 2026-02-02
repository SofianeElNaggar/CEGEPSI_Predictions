import csv

# Chemin du fichier CSV en entrée
csv_path = "../../dataset/meteogc/Station Inventory EN 2.csv"

# Chemin du fichier TXT en sortie
txt_path = "../../dataset/meteogc/stations2.txt"

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)

    with open(txt_path, "w", encoding="utf-8") as txtfile:
        for row in reader:
            if len(row) >= 4:  # Vérifie que la colonne 4 existe
                txtfile.write(row[3] + "\n")  # Colonne 4 = index 3
