import csv
import tempfile
import os

input_csv_path = '../../dataset/OMDS-CTD-meteogc-data.csv'

new_header = [
    "time(UTC)",
    "latitude(degrees_north)",
    "longitude(degrees_east)",
    "depth(m),pressure(dbar)",
    "par_downwelling(MicroEinsteins.m-2.s-1)",
    "par_upwelling(MicroEinsteins.m-2.s-1)",
    "temperature(°C)",
    "chlorophyll(mg.m-3)",
    "turbidity(NTU)",
    "pH(pH)",
    "salinity(PSS-78)",
    "dissolved_oxygen(ml.l-1)",
    "tide_range(m)",
    "MaxTemp(°C)",
    "MinTemp(°C)",
    "MeanTemp(°C)",
    "HeatDegDays(°C)",
    "CoolDegDays(°C)",
    "TotalPrecip(mm)",
    "DirofMaxGust(10s.deg)",
    "SpdofMaxGust(km/h)"
]

# Créer un fichier temporaire
with tempfile.NamedTemporaryFile('w', delete=False, newline='', encoding='utf-8') as tmpfile:
    with open(input_csv_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        writer = csv.writer(tmpfile)

        # Remplacer la première ligne par le nouvel en-tête
        try:
            next(reader)  # ignorer l'ancien header
        except StopIteration:
            pass  # fichier vide, pas de problème

        writer.writerow(new_header)

        # Copier le reste du fichier ligne par ligne
        for row in reader:
            writer.writerow(row)

# Remplacer le fichier original par le fichier temporaire
os.replace(tmpfile.name, input_csv_path)

print(f"Le CSV original a été mis à jour avec le nouvel en-tête.")
