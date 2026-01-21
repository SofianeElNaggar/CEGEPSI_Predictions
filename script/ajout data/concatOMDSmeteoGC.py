#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
join_meteo_fixed_paths.py
Associe à chaque ligne du CSV de mesures la station météo la plus proche et
les données météo du même jour (si disponibles).
Les chemins sont fixés directement dans le code.
"""

import os
import csv
import math
from datetime import datetime
from collections import defaultdict

# -------------------------
# Chemins à modifier ici
# -------------------------
INPUT_MEASURES = "../../dataset/Tide/Tide OMDS-CTD data.csv"
STATIONS_DIR = "../../dataset/meteogc"
OUTPUT_CSV = "../../dataset/meteogc-OMDS-dataset.csv"

# -------------------------
# Fonctions utilitaires
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Distance haversine en km"""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def parse_iso_date_to_YYYYMMDD_and_hour(iso_str):
    """Parse ISO timestamp en (YYYYMMDD, hour int)"""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y%m%d"), dt.hour
    except Exception:
        return None, None

# -------------------------
# Charger les stations
# -------------------------
def load_stations(stations_root):
    """
    Parcourt stations_root, trouve tous les CSV et charge :
      - metadata: station_id, lat, lon
      - data index: dict[station_id][YYYYMMDD] -> list of (hour_int, row_dict)
    Retourne (stations_meta, station_day_index, station_meteo_headers)

    Note :
      - Les colonnes gardent exactement leurs noms d'origine.
      - Les colonnes dupliquées (ex: "Flag") sont conservées telles quelles.
      - Les valeurs sont conservées telles quelles (string/int/float), sauf lat/lon pour le calcul de distance.
    """
    import os
    import csv
    from collections import defaultdict

    stations_meta = {}
    station_day_index = {}
    meteo_headers = None
    files_found = 0

    print(f"[DEBUG] Lecture des fichiers stations dans : {stations_root}")
    for root, dirs, files in os.walk(stations_root):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            path = os.path.join(root, fn)
            files_found += 1
            if files_found % 50 == 0:
                print(f"[DEBUG] {files_found} fichiers de stations trouvés (jusqu'à présent)")

            try:
                with open(path, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    # lire les deux premières lignes d'info
                    try:
                        info1 = next(reader)
                        info2 = next(reader)
                    except StopIteration:
                        print(f"[WARN] Fichier station {path} trop court, ignoré.")
                        continue

                    # metadata E2,F2,G2 -> indices 4,5,6
                    station_id = info2[4].strip() if len(info2) > 4 else os.path.splitext(fn)[0]

                    try:
                        latf = float(info2[5].strip()) if len(info2) > 5 else None
                        lonf = float(info2[6].strip()) if len(info2) > 6 else None
                    except Exception:
                        print(f"[WARN] Lat/lon invalide pour {path}, ignorée")
                        continue  # on ignore cette station si lat/lon non valides

                    # lire l'entête (3ème ligne)
                    try:
                        header = next(reader)
                    except StopIteration:
                        print(f"[WARN] Fichier station {path} sans header/data, ignoré.")
                        continue

                    # Ne pas modifier les noms de colonnes, même si doublons
                    header_orig = [h.strip() for h in header]
                    try:
                        date_col_idx = header_orig.index("Date/Time")
                    except ValueError:
                        print(f"[WARN] Colonne date introuvable dans {path}, ignorée.")
                        continue

                    stations_meta[station_id] = {'lat': latf, 'lon': lonf, 'path': path}
                    day_index = defaultdict(list)

                    # lire toutes les lignes restantes
                    for row in reader:
                        if not row or all(cell.strip() == "" for cell in row):
                            continue
                        if date_col_idx >= len(row):
                            continue
                        date_field = row[date_col_idx].strip()
                        if len(date_field) < 8:
                            continue
                        day = date_field[:8]
                        try:
                            hour = int(date_field[-2:])
                        except Exception:
                            hour = 0
                        # Construire dict col->valeur, garder exactement le nom et la valeur
                        # Même si des colonnes ont le même nom, on garde tel quel
                        row_dict = {header_orig[i]: row[i].strip() if i < len(row) else "" for i in
                                    range(len(header_orig))}
                        day_index[day].append((hour, row_dict))

                    station_day_index[station_id] = day_index

                    if meteo_headers is None:
                        # On garde les noms originaux, même si doublons
                        meteo_headers = header_orig

            except Exception as e:
                print(f"[ERROR] Erreur lecture fichier station {path}: {e}")

    print(f"[DEBUG] Fini lecture stations. {len(stations_meta)} stations chargées à partir de {files_found} fichiers.")
    return stations_meta, station_day_index, meteo_headers


# -------------------------
# Traitement des mesures
# -------------------------
def process_measures():
    stations_meta, station_day_index, meteo_headers = load_stations(STATIONS_DIR)
    if not stations_meta:
        print("[ERROR] Aucune station chargée. Arrêt.")
        return

    meteo_headers_prefixed = [h for h in meteo_headers if h!="Year Month Day Hour (YYYYMMDDHH)"]

    with open(INPUT_MEASURES, newline='', encoding='utf-8') as fin, \
         open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        title_line = next(reader)
        try:
            units_line = next(reader)
        except StopIteration:
            units_line = []

        new_header = title_line + ["station_id","station_lat","station_lon"] + meteo_headers_prefixed
        writer.writerow(new_header)
        writer.writerow(units_line + [""]* (3+len(meteo_headers_prefixed)))

        stations_list = [(sid,md['lat'],md['lon']) for sid,md in stations_meta.items()]

        for i,row in enumerate(reader, start=1):
            if i%100000==0:
                print(f"[DEBUG] {i} lignes traitées")
            if not row:
                writer.writerow(row + [""]* (3+len(meteo_headers_prefixed)))
                continue
            time_field = row[0].strip()
            day_str,hour_meas = parse_iso_date_to_YYYYMMDD_and_hour(time_field)
            if day_str is None:
                writer.writerow(row + [""]* (3+len(meteo_headers_prefixed)))
                continue
            try:
                lat_val = float(row[1].strip())
                lon_val = float(row[2].strip())
            except:
                writer.writerow(row + [""]* (3+len(meteo_headers_prefixed)))
                continue

            # station la plus proche
            nearest_sid, nearest_dist = None, float("inf")
            for sid,slat,slon in stations_list:
                d = haversine(lat_val, lon_val, slat, slon)
                if d<nearest_dist:
                    nearest_dist=d
                    nearest_sid=sid

            day_entries = station_day_index.get(nearest_sid,{}).get(day_str)
            if not day_entries:
                st_meta = stations_meta.get(nearest_sid,{})
                writer.writerow(row + [nearest_sid,str(st_meta.get('lat','')),str(st_meta.get('lon',''))] + [""]*len(meteo_headers_prefixed))
                continue

            best_entry = min(day_entries,key=lambda x: abs(x[0]-hour_meas))[1]
            st_meta = stations_meta.get(nearest_sid,{})
            meteo_values = [best_entry.get(h,"") for h in meteo_headers if h!="Year Month Day Hour (YYYYMMDDHH)"]
            writer.writerow(row + [nearest_sid,str(st_meta.get('lat','')),str(st_meta.get('lon',''))] + meteo_values)

    print(f"[INFO] Traitement terminé. Fichier écrit : {OUTPUT_CSV}")

# -------------------------
# Main
# -------------------------
if __name__=="__main__":
    process_measures()
