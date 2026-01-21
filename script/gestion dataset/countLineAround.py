import pandas as pd
import numpy as np

# Coordonnées de Sept-Îles
sept_iles_lat = 50.2000
sept_iles_lon = -66.3756

# Date limite (ignorer avant cette date)
DATE_LIMIT = pd.Timestamp("2015-01-01 00:00:00+00:00")

DEPTH_MIN = 273
DEPTH_MAX = 340


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


df = pd.read_parquet("../../dataset/OMDS-CTD-meteogc-data.parquet")

# Convertir colonnes numériques
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["depth"] = pd.to_numeric(df["depth"], errors="coerce")   # 🔥 OBLIGATOIRE

# Convertir time
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Retirer lignes invalides
df = df.dropna(subset=["latitude", "longitude", "time"])

# FILTRES
df = df[df["time"] >= DATE_LIMIT]
df = df[(df["depth"] >= DEPTH_MIN) & (df["depth"] <= DEPTH_MAX)]

# Distance
df["distance_km"] = haversine(df["latitude"], df["longitude"], sept_iles_lat, sept_iles_lon)

df_filtered = df[df["distance_km"] <= 100]

count_lines = len(df_filtered)

nan_counts = df_filtered.isna().sum()
nan_percent = (nan_counts / count_lines) * 100

nan_report = pd.DataFrame({
    "NaN_count": nan_counts,
    "NaN_percent": nan_percent
})

print("Nombre de lignes à moins de 100 km :", count_lines)
print("\nRésumé NaN par colonne :")
print(nan_report)
