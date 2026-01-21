#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrichit csv1.csv avec les variables météo de la station la plus proche
au moment de la mesure.

Hypothèses (à adapter si besoin) :
 - csv1.csv dans le répertoire courant
 - csv2.csv dans le répertoire courant
 - dossier 'stations/' contenant un sous-dossier par Station ID (ex: stations/27803/)
 - fichiers annuels nommés exactement data_{StationID}_{YYYY}.csv

Dépendances : pandas, numpy
Installe avec: pip install pandas numpy
"""

import os
import csv
import math
from collections import OrderedDict
from datetime import datetime
import pandas as pd
import numpy as np

# -------- CONFIG (modifie si nécessaire) ----------
CSV1_PATH = "../../dataset/Tide/Tide OMDS-CTD data.csv"
CSV2_PATH = "../../dataset/meteogc/Station Inventory EN.csv"
STATIONS_DIR = ("../../dataset/meteogc/v2")
OUTPUT_PATH = "../../dataset/OMDS-CTD-meteogc-v2-data.csv"

# taille du bloc pour vectoriser la recherche de station la plus proche
BLOCK_SIZE = 5000

# cache pour fichiers station-year (LRU)
CACHE_MAX_ITEMS = 100

# colonnes météo à extraire (doivent être exactement les noms dans les fichiers station-year)
METEO_COLUMNS = [
    "Max Temp (°C)",
    "Min Temp (°C)",
    "Mean Temp (°C)",
    "Heat Deg Days (°C)",
    "Cool Deg Days (°C)",
    "Total Precip (mm)",
    "Dir of Max Gust (10s deg)",
    "Spd of Max Gust (km/h)",
]

# --------------------------------------------------

def haversine_vec(lat1, lon1, lats2, lons2):
    """
    calcule distance (km) entre (lat1,lon1) et arrays lats2,lons2 (en degrés)
    vectorisé pour performance.
    """
    # convert to radians
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lats2)
    lon2r = np.radians(lons2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371.0  # Earth radius km
    return R * c

class StationYearCache:
    """
    LRU cache pour DataFrame station-year.
    stocke pandas.DataFrame indexé sur Date/Time (to_datetime)
    """
    def __init__(self, max_items=CACHE_MAX_ITEMS):
        self.max_items = max_items
        self.cache = OrderedDict()

    def _make_key(self, station_id, year):
        return f"{station_id}::{year}"

    def get(self, station_id, year):
        key = self._make_key(station_id, year)
        if key in self.cache:
            # move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        df = self._load(station_id, year)
        if df is None:
            return None
        self.cache[key] = df
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return df

    def _load(self, station_id, year):
        # chemin attendu : stations/{station_id}/data_{station_id}_{year}.csv
        folder = os.path.join(STATIONS_DIR, str(station_id))
        fname = f"data_{station_id}_{year}.csv"
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            # pas de fichier pour cette station/année
            return None
        # Lire avec pandas (col Date/Time parsed)
        try:
            # On lit toutes les colonnes demandées + 'Date/Time' si présent
            usecols = None  # laisser pandas déduire pour éviter d'échouer si colonnes manquent
            df = pd.read_csv(path, parse_dates=["Date/Time"], dayfirst=False, low_memory=True)
            # normaliser l'index sur Date/Time pour lookup
            if "Date/Time" in df.columns:
                df = df.set_index(pd.to_datetime(df["Date/Time"], errors="coerce"))
            else:
                # si pas de colonne 'Date/Time', tenter de trouver une colonne similaire
                # on retourne None pour indiquer qu'on ne peut pas matcher
                return None
            # ne garder que les colonnes météo demandées (si elles existent)
            available = [c for c in METEO_COLUMNS if c in df.columns]
            if len(available) == 0:
                # pas de colonnes utiles
                return None
            df = df[available]
            return df
        except Exception as e:
            # en cas d'erreur de parsing, renvoyer None (pas catastrophique)
            print(f"Warning: impossible de lire {path}: {e}")
            return None

def find_nearest_station_block(lat_block, lon_block, station_lats, station_lons):
    """
    lat_block, lon_block : arrays shape (N,)
    station_lats, station_lons : arrays shape (M,)
    retourne indices des stations les plus proches pour chaque point
    """
    # pour chaque point du bloc, on calcule distances vectorisées vers toutes les stations
    # et on prend argmin. On le fait en vectorisant sur le blocs pour éviter boucle python par ligne.
    N = len(lat_block)
    M = len(station_lats)
    # Pour grande taille de M, on peut faire par sous-blocs pour mémoire - mais M est souvent raisonnable
    nearest_idx = np.empty(N, dtype=np.int32)
    for i in range(N):
        d = haversine_vec(lat_block[i], lon_block[i], station_lats, station_lons)
        nearest_idx[i] = int(np.argmin(d))
    return nearest_idx

def safe_to_datetime(s):
    try:
        # pandas to_datetime est robuste
        dt = pd.to_datetime(s, utc=True, errors='coerce')
        return dt
    except Exception:
        return pd.NaT

def main():
    # -> charger stations (csv2)
    if not os.path.isfile(CSV2_PATH):
        raise FileNotFoundError(f"{CSV2_PATH} introuvable. Place csv2.csv dans le répertoire courant ou modifie CSV2_PATH.")
    stations_df = pd.read_csv(CSV2_PATH, low_memory=True)
    # normaliser noms possibles de colonnes (différents fichiers peuvent varier)
    # on cherche 'Station ID' et 'Latitude (Decimal Degrees)' / 'Longitude (Decimal Degrees)'
    col_station = None
    col_lat = None
    col_lon = None
    for c in stations_df.columns:
        lc = c.strip().lower()
        if "station" in lc and "id" in lc:
            col_station = c
        if "latitude" in lc and "decimal" in lc:
            col_lat = c
        if "longitude" in lc and "decimal" in lc:
            col_lon = c
    if col_station is None:
        # essayer heuristique
        possible = [c for c in stations_df.columns if "station" in c.lower()]
        if possible:
            col_station = possible[0]
    if col_lat is None:
        possible = [c for c in stations_df.columns if "latitude" in c.lower()]
        col_lat = possible[0] if possible else None
    if col_lon is None:
        possible = [c for c in stations_df.columns if "longitude" in c.lower()]
        col_lon = possible[0] if possible else None

    if col_station is None or col_lat is None or col_lon is None:
        raise ValueError("Impossible de trouver les colonnes Station ID / Latitude (Decimal Degrees) / Longitude (Decimal Degrees) dans csv2.csv. Vérifie les noms de colonnes.")

    stations_df = stations_df[[col_station, col_lat, col_lon]].dropna()
    stations_df[col_lat] = pd.to_numeric(stations_df[col_lat], errors='coerce')
    stations_df[col_lon] = pd.to_numeric(stations_df[col_lon], errors='coerce')
    stations_df = stations_df.dropna(subset=[col_lat, col_lon])
    station_ids = stations_df[col_station].astype(str).tolist()
    station_lats = stations_df[col_lat].to_numpy(dtype=float)
    station_lons = stations_df[col_lon].to_numpy(dtype=float)

    if len(station_ids) == 0:
        raise ValueError("Aucune station valide trouvée dans csv2.csv.")

    print(f"Chargé {len(station_ids)} stations depuis {CSV2_PATH}.")

    cache = StationYearCache(max_items=CACHE_MAX_ITEMS)

    # Préparer lecture/écriture CSV1 (ligne par ligne)
    if not os.path.isfile(CSV1_PATH):
        raise FileNotFoundError(f"{CSV1_PATH} introuvable. Place csv1.csv dans le répertoire courant ou modifie CSV1_PATH.")

    # lecture entête de csv1 pour conserver les colonnes originales
    with open(CSV1_PATH, newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        original_fieldnames = reader.fieldnames if reader.fieldnames is not None else []
    # champs de sortie = originaux + METEO_COLUMNS
    out_fieldnames = list(original_fieldnames) + METEO_COLUMNS

    # on ouvre input et output et on lit par blocs pour vectoriser la recherche de station
    with open(CSV1_PATH, newline='', encoding='utf-8') as fin, \
         open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
        writer.writeheader()

        block_rows = []
        total = 0
        for row in reader:
            block_rows.append(row)
            if len(block_rows) >= BLOCK_SIZE:
                process_block(block_rows, station_ids, station_lats, station_lons, cache, writer)
                total += len(block_rows)
                print(f"Traitées ~{total} lignes...")
                block_rows = []
        # dernier bloc
        if block_rows:
            process_block(block_rows, station_ids, station_lats, station_lons, cache, writer)
            total += len(block_rows)
        print(f"Terminé. {total} lignes traitées. Résultat écrit dans {OUTPUT_PATH}")

def process_block(block_rows, station_ids, station_lats, station_lons, cache, writer):
    """
    Traite un bloc de lignes (liste de dicts), trouve la station la plus proche
    pour chaque ligne et enrichit avec les colonnes météo, puis écrit avec writer.
    """
    n = len(block_rows)
    lats = np.empty(n, dtype=float)
    lons = np.empty(n, dtype=float)
    times = [None] * n

    # Extraire lat/lon/time de chaque ligne (colonnes attendues 'latitude' et 'longitude' / 'time')
    for i, r in enumerate(block_rows):
        # essayer plusieurs noms possibles de colonnes (selon CSV1)
        lat = None
        lon = None
        if 'latitude' in r:
            lat = r.get('latitude')
        elif 'Latitude' in r:
            lat = r.get('Latitude')
        elif 'lat' in r:
            lat = r.get('lat')
        if 'longitude' in r:
            lon = r.get('longitude')
        elif 'Longitude' in r:
            lon = r.get('Longitude')
        elif 'lon' in r:
            lon = r.get('lon')

        try:
            lats[i] = float(lat) if lat not in (None, "") else np.nan
        except Exception:
            lats[i] = np.nan
        try:
            lons[i] = float(lon) if lon not in (None, "") else np.nan
        except Exception:
            lons[i] = np.nan

        # time colonne
        t = None
        if 'time' in r:
            t = r.get('time')
        elif 'Time' in r:
            t = r.get('Time')
        elif 'date' in r:
            t = r.get('date')
        times[i] = safe_to_datetime(t)  # pandas Timestamp or NaT

    # Pour les lignes sans lat/lon -> on met des NaN distances -> résultat None
    # trouver nearest station pour chaque ligne non-NaN
    valid_mask = ~np.isnan(lats) & ~np.isnan(lons)
    nearest_indices = np.full(n, -1, dtype=int)
    if valid_mask.any():
        valid_idxs = np.where(valid_mask)[0]
        # vectoriser: pour chaque point valide on calcule distances aux stations
        # boucle sur les points valides (on pourrait chunker si très grand) ; BLOCK_SIZE contrôle ça
        for idx in valid_idxs:
            d = haversine_vec(lats[idx], lons[idx], station_lats, station_lons)
            nearest_indices[idx] = int(np.argmin(d))

    # pour chaque ligne, récupérer la valeur météo et écrire
    for i, r in enumerate(block_rows):
        meteo_values = {c: "" for c in METEO_COLUMNS}
        if nearest_indices[i] >= 0:
            station_idx = nearest_indices[i]
            station_id = station_ids[station_idx]
            dt = times[i]
            if pd.isna(dt):
                # pas de timestamp parseable -> on ne peut pas chercher
                pass
            else:
                year = dt.year
                df = cache.get(station_id, year)
                if df is not None:
                    # chercher correspondance exacte sur datetime (index datetime)
                    matches = df.index == dt.to_datetime64() if hasattr(dt, "to_datetime64") else df.index == pd.to_datetime(dt)
                    # use pandas boolean index
                    matched_rows = df[matches]
                    if matched_rows.shape[0] == 0:
                        # essayer correspondance sur date seulement
                        try:
                            date_only = dt.date()
                            # index could be timezone-aware -> compare normalized dates
                            mask_date = df.index.date == date_only
                            matched_rows = df[mask_date]
                        except Exception:
                            matched_rows = df.iloc[0:0]  # empty
                    if matched_rows.shape[0] > 0:
                        # prendre la première correspondance
                        first = matched_rows.iloc[0]
                        for col in METEO_COLUMNS:
                            if col in matched_rows.columns:
                                val = first.get(col)
                                # convertir NaN->"" et garder str
                                if pd.isna(val):
                                    meteo_values[col] = ""
                                else:
                                    meteo_values[col] = str(val)
        # écrire ligne enrichie
        out_row = {}
        # conserver l'ordre de original csv
        for fn in writer.fieldnames:
            if fn in r:
                out_row[fn] = r[fn]
            elif fn in METEO_COLUMNS:
                out_row[fn] = meteo_values.get(fn, "")
            else:
                out_row[fn] = ""
        writer.writerow(out_row)

if __name__ == "__main__":
    main()
