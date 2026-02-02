import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# ==============================
# PARAMÈTRES À ADAPTER
# ==============================
CSV_PATH = "../../dataset/OMDS-CTD-meteogc-data.csv"
COLUMN_NAME = "depth"

# Coordonnées de Sept-Îles
sept_iles_lat = 50.2000
sept_iles_lon = -66.3756

# Filtre temps
DATE_LIMIT = pd.Timestamp("2015-01-01 00:00:00+00:00")

# Distance max
MAX_DISTANCE_KM = 100
# ==============================


# Fonction Haversine (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# Lecture CSV
df = pd.read_csv(CSV_PATH)

# Conversion des types
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df[COLUMN_NAME] = pd.to_numeric(df[COLUMN_NAME], errors="coerce")
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Suppression des lignes invalides
df = df.dropna(subset=["latitude", "longitude", "time"])

# Filtre temps
df = df[df["time"] >= DATE_LIMIT]

# Calcul de distance
df["distance_km"] = haversine(
    df["latitude"], df["longitude"],
    sept_iles_lat, sept_iles_lon
)

# Filtre distance
df = df[df["distance_km"] <= MAX_DISTANCE_KM]

# Extraction des valeurs de la colonne choisie
values = df[COLUMN_NAME].dropna()

if len(values) == 0:
    raise ValueError("Aucune valeur exploitable après filtrage.")

# Bornes des classes
min_val = math.floor(values.min())
max_val = math.ceil(values.max())

print("Valeur max dans la zone filtrée :", values.max())

bins = range(min_val, max_val + 1)

# Histogramme
plt.hist(values, bins=bins, edgecolor="black")
plt.xlabel(COLUMN_NAME)
plt.ylabel("Fréquence")
plt.title(f"Histogramme de '{COLUMN_NAME}' filtré par temps + distance")
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()
