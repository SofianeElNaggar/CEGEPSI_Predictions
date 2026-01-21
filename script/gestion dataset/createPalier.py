import pandas as pd
import numpy as np
from pathlib import Path

# ---------- CONFIG ----------
path = "../../dataset/OMDS-CTD-meteogc-data.csv"
depth_col = "depth"
n_bins = 15
round_to = 1.0

# Filtres ajoutés
DATE_LIMIT = pd.Timestamp("2015-01-01 00:00:00+00:00")
IGNORE_LAT = 50.2011
IGNORE_LON = -66.3821
MAX_DISTANCE_KM = 100

# Filtre depth supplémentaire
DEPTH_MIN = None   # ex: 0
DEPTH_MAX = None   # ex: 2
# -----------------------------

# Fonction Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    return 2 * R * np.arcsin(
        np.sqrt(
            np.sin((lat2 - lat1) / 2) ** 2 +
            np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
    )

# Coordinates of Sept-Îles
sept_iles_lat = 50.2000
sept_iles_lon = -66.3756

# ---------- LOAD ----------
df = pd.read_csv(path)

# ---------- VALIDATE ----------
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")

df = df.dropna(subset=["latitude", "longitude", "time", depth_col])

# ---------- APPLY FILTERS ----------

# 1) Filtre temps
df = df[df["time"] >= DATE_LIMIT]

# 2) Filtre coordonnée à ignorer
df = df[~((df["latitude"] == IGNORE_LAT) & (df["longitude"] == IGNORE_LON))]

# 3) Calcul distance
df["distance_km"] = haversine(
    df["latitude"], df["longitude"],
    sept_iles_lat, sept_iles_lon
)

# 4) Filtre distance
df = df[df["distance_km"] <= MAX_DISTANCE_KM]

# 5) Filtre depth min/max
if DEPTH_MIN is not None:
    df = df[df[depth_col] >= DEPTH_MIN]
if DEPTH_MAX is not None:
    df = df[df[depth_col] <= DEPTH_MAX]

# ---------- BINS ----------
depths = df[depth_col].dropna().astype(float)

if depths.empty:
    raise ValueError("Plus aucune donnée valide après filtrage !")

min_d = depths.min()
max_d = depths.max()

# --- Quantiles pour bins équilibrés ---
quantiles = np.linspace(0, 1, n_bins + 1)
edges = depths.quantile(quantiles).to_numpy()

# --- Round edges ---
def round_edges(arr, step):
    return np.round(arr / step) * step

rounded = round_edges(edges, round_to)

# --- Ensure strictly increasing ---
def enforce_strict_increasing(b, step, min_val, max_val):
    b = b.copy().astype(float)
    b[0] = np.round(min_val / step) * step
    b[-1] = np.round(max_val / step) * step

    if b[0] >= b[-1]:
        b[0] = min_val
        b[-1] = max_val

    for i in range(1, len(b)):
        if b[i] <= b[i - 1]:
            b[i] = b[i - 1] + step

    if b[-1] > max_val:
        shift = b[-1] - max_val
        b -= shift

    return b

final_edges = enforce_strict_increasing(rounded, round_to, min_d, max_d)

if not np.all(np.diff(final_edges) > 0):
    final_edges = edges.copy()

# ----- Create labels -----
interval_labels = []
for i in range(len(final_edges) - 1):
    a = final_edges[i]
    b = final_edges[i + 1]
    a_str = f"{int(a)}" if round_to >= 1 else f"{a:.2f}"
    b_str = f"{int(b)}" if round_to >= 1 else f"{b:.2f}"
    label = f"[{a_str} - {b_str}" + (")" if i < len(final_edges)-2 else "]") + " m"
    interval_labels.append(label)

# ----- Count -----
cat = pd.cut(depths, bins=final_edges, right=False, include_lowest=True)
counts = cat.value_counts(sort=False)

summary = pd.DataFrame({
    "interval": interval_labels,
    "count": counts.to_numpy()
})

# ---------- Print result ----------
print("\nRÉCAPITULATIF DES PALIERS (APRÈS FILTRES) :\n")
print(summary.to_string(index=False))

print("\nTotal mesures retenues :", len(depths))
print("Somme des counts :", summary["count"].sum())
