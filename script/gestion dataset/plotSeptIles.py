import pandas as pd
import numpy as np
from math import radians, sin, cos, asin, sqrt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------------------------------
# 1) Paramètres
# ----------------------------------------------------

PARQUET_PATH = "../../dataset/OMDS-CTD-meteogc-data.parquet"

SEPT_ILES_LAT = 50.2
SEPT_ILES_LON = -66.38
RADIUS_KM = 200

# ----------------------------------------------------
# 2) Fonction haversine
# ----------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# ----------------------------------------------------
# 3) Chargement des données
# ----------------------------------------------------
df = pd.read_parquet(PARQUET_PATH)

df['time (UTC)'] = pd.to_datetime(df['time (UTC)'])

# ----------------------------------------------------
# 4) Filtrage spatial
# ----------------------------------------------------
distances = df.apply(
    lambda r: haversine(SEPT_ILES_LAT, SEPT_ILES_LON,
                        r['latitude (degrees north)'],
                        r['longitude (degrees_east)']),
    axis=1
)

df_filtered = df[distances <= RADIUS_KM].copy()
print(f"Points retenus : {len(df_filtered)}")

# ----------------------------------------------------
# 5) Colonnes à tracer
# ----------------------------------------------------
exclude = ['time (UTC)', 'latitude (degrees north)', 'longitude (degrees_east)']
columns_to_plot = [c for c in df_filtered.columns if c not in exclude]

# ----------------------------------------------------
# 6) Génération du PDF
# ----------------------------------------------------
output_pdf = "plots_variables.pdf"

with PdfPages(output_pdf) as pdf:
    for col in columns_to_plot:

        # Conversion en numérique (important !)
        y = pd.to_numeric(df_filtered[col], errors="coerce")

        # Retirer les NaN
        valid = df_filtered[['time (UTC)']].copy()
        valid[col] = y
        valid = valid.dropna()

        # Si colonne vide → skip
        if valid.empty:
            print(f"⚠️  Colonne vide après nettoyage : {col} (skip)")
            continue

        plt.figure(figsize=(10,5))
        plt.plot(valid['time (UTC)'], valid[col])
        plt.title(col)
        plt.xlabel("Time (UTC)")
        plt.ylabel(col)
        plt.grid(True)
        plt.tight_layout()

        pdf.savefig()
        plt.close()

print(f"PDF généré : {output_pdf}")
