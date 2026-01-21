# tide_fixed.py
# Python 3.8+
# dépendances: requests, pandas
# pip install requests pandas

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import sys

def daterange(start_date, end_date):
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)

def split_months(start, end):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        chunk_end = min(end, nxt - timedelta(days=1))
        yield max(cur, start), chunk_end
        cur = nxt

def _extract_list_from_json(j):
    """
    Normalise différentes formes de réponse JSON en une liste d'objets.
    NOAA habituellement renvoie une liste, ou {"predictions": [...]}.
    """
    if j is None:
        return []
    if isinstance(j, list):
        return j
    if isinstance(j, dict):
        # Common keys to look for
        for key in ("predictions", "data", "results", "items"):
            if key in j and isinstance(j[key], list):
                return j[key]
        # Sometimes API returns {"error": "..."}
        if "error" in j:
            raise RuntimeError("API returned error: " + str(j.get("error")))
        # fallback: try to find first list value
        for v in j.values():
            if isinstance(v, list):
                return v
    # unknown shape
    return []

def _find_time_and_value_keys(obj):
    """
    Retourne (time_key, value_key) en inspectant les clés d'un objet.
    """
    time_keys = ['t', 'time', 'dateTime', 'datetime', 'timestamp']
    value_keys = ['v', 'value', 'height', 'y']
    keys = set(obj.keys())
    tk = next((k for k in time_keys if k in keys), None)
    vk = next((k for k in value_keys if k in keys), None)
    return tk, vk

def fetch_hourly_noaa(station, chunk_start, chunk_end, sleep_if_rate=True):
    """
    Fetch hourly tide predictions from NOAA CO-OPS API for given date chunk.
    Returns pandas.DataFrame with columns ['t','v'] where v=float height (units=metric).
    """
    base = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "product": "predictions",
        "application": "tide-daily-range-script",
        "station": station,
        "begin_date": chunk_start.strftime("%Y%m%d"),
        "end_date": chunk_end.strftime("%Y%m%d"),
        "datum": "MLLW",
        "units": "metric",
        "time_zone": "gmt",
        "interval": "h",
        "format": "json"
    }
    r = requests.get(base, params=params, timeout=60)
    if r.status_code != 200:
        # affiche le texte complet pour debug
        raise RuntimeError(f"NOAA API HTTP {r.status_code}: {r.text[:1000]}")
    try:
        j = r.json()
    except ValueError:
        # réponse non-JSON
        raise RuntimeError(f"Réponse non-JSON: {r.text[:1000]}")

    # Normalise
    items = _extract_list_from_json(j)
    if not items:
        # pour debug : montre le JSON reçu
        raise RuntimeError(f"Réponse JSON sans liste d'items pour {chunk_start} -> {chunk_end} : {j}")

    # detect keys
    first = items[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"Format inattendu des items (attendu dict) : {first}")

    time_key, value_key = _find_time_and_value_keys(first)
    if time_key is None or value_key is None:
        # log pour debug : show keys seen
        raise RuntimeError(f"Impossible d'identifier champs temps/valeur. Clés trouvées: {list(first.keys())}. JSON reçu (tronc): {str(j)[:800]}")

    # build df
    df = pd.DataFrame(items)
    # rename to canonical 't' and 'v'
    df = df.rename(columns={time_key: 't', value_key: 'v'})
    # parse datetime and numeric
    df['t'] = pd.to_datetime(df['t'], utc=True, errors='coerce')
    df['v'] = pd.to_numeric(df['v'], errors='coerce')
    # drop rows where parse failed
    df = df.dropna(subset=['t'])
    return df[['t','v']]

def compute_daily_range_from_df(df, local_tz=None):
    """Input: df with column 't' (datetime) and 'v' (float). Returns DataFrame with date and range."""
    if df.empty:
        return pd.DataFrame(columns=['date','max','min','range_m'])
    df = df.copy()
    if local_tz:
        # convert to timezone-aware local if requested (pytz or zoneinfo name)
        try:
            df['t'] = df['t'].dt.tz_convert(local_tz)
        except Exception:
            pass
    df['date'] = df['t'].dt.date
    grp = df.groupby('date')['v'].agg(['max','min']).reset_index()
    grp['range_m'] = grp['max'] - grp['min']
    return grp[['date','max','min','range_m']]

def make_daily_tidal_ranges_noaa(station, start_date, end_date, out_csv="daily_tidal_range.csv"):
    all_rows = []
    for chunk_start, chunk_end in split_months(start_date, end_date):
        print(f"Fetching {chunk_start.date()} -> {chunk_end.date()}")
        try:
            df_chunk = fetch_hourly_noaa(station, chunk_start, chunk_end)
        except Exception as e:
            # log l'erreur mais continue (on remplira plus tard avec NaN)
            print(f"Erreur lors de la récupération {chunk_start.date()}->{chunk_end.date()}: {e}", file=sys.stderr)
            # continue sans ajouter de chunk pour cette période
            continue
        daily = compute_daily_range_from_df(df_chunk)
        all_rows.append(daily)
        time.sleep(0.8)
    if all_rows:
        result = pd.concat(all_rows, ignore_index=True)
    else:
        result = pd.DataFrame(columns=['date','max','min','range_m'])
    # ensure full requested date coverage
    full_index = pd.DataFrame({'date':[d.date() for d in daterange(start_date, end_date)]})
    merged = full_index.merge(result, on='date', how='left')
    merged.to_csv(out_csv, index=False)
    print(f"Saved {len(merged)} daily records to {out_csv}")
    return merged

if __name__ == "__main__":
    # Remplace par la station NOAA que tu utilises
    station = "8518750"   # <-- à adapter
    start = datetime(2000,1,1)
    end   = datetime(2025,11,24)
    df = make_daily_tidal_ranges_noaa(station, start, end, out_csv="daily_tidal_range_noaa.csv")
    print(df.head(10))
