# merge_range_m.py
# Script : ajoute la colonne 'range_m' du CSV B dans le CSV A.
# - Définitions des chemins de fichiers en haut du fichier
# - Si une date de B n'existe pas dans A, on crée une ligne (autres colonnes = NaN),
#   la colonne date est au format ISO avec "Z" (ex: 2000-01-01T00:00:00Z) pour être cohérent avec A.

import pandas as pd
import numpy as np

# ---------- CONFIG -----------
input_csv_a = "../../dataset/Clear OMDS-CTD data.csv"        # <-- mettre le chemin du csv A ici
input_csv_b = "../../dataset/daily_tidal_range_noaa.csv"      # <-- mettre le chemin du csv B ici
output_csv = "../../dataset/Tided Updated OMDS-CTD data.csv" # <-- nom du csv de sortie

# Si vous connaissez déjà le nom exact des colonnes de date, mettez-les ici.
# Si left as None, le script va essayer de détecter automatiquement.
date_col_a = "time"   # ex: "timestamp" or "date_time" ; None = auto-detect
date_col_b = "date"   # ex: "date" ; None = auto-detect
# -------------------------------

def detect_date_col(df):
    """Détecte la colonne la plus vraisemblable contenant des dates."""
    candidates = [c for c in df.columns if any(tok in c.lower() for tok in ("date","time","timestamp"))]
    if candidates:
        return candidates[0]
    # fallback: tester les premières colonnes en essayant de parser
    for c in df.columns[:3]:
        try:
            pd.to_datetime(df[c].dropna().iloc[:20], errors='raise')
            return c
        except Exception:
            continue
    # sinon renvoyer la première colonne
    return df.columns[0]

# --- Lire CSV A ---
df_a = pd.read_csv(input_csv_a, dtype=str)  # lire en str d'abord pour éviter erreurs de parsing
if date_col_a is None:
    date_col_a = detect_date_col(df_a)
# parser A : format attendu comme 2000-08-24T14:35:46Z (ISO zulu)
# on force utc et coerce les erreurs en NaT
df_a[date_col_a] = pd.to_datetime(
    df_a[date_col_a],
    format="%Y-%m-%dT%H:%M:%SZ",
    utc=True,
    errors='coerce'
)


# --- Lire CSV B ---
df_b = pd.read_csv(input_csv_b, dtype=str)
if date_col_b is None:
    date_col_b = detect_date_col(df_b)
# parser B : format attendu 'YYYY-MM-DD'
df_b[date_col_b] = pd.to_datetime(df_b[date_col_b], errors='coerce').dt.tz_localize(None)  # naive dates

# s'assurer que la colonne 'range_m' existe dans B
if 'range_m' not in df_b.columns:
    raise KeyError("La colonne 'range_m' n'a pas été trouvée dans le CSV B. Vérifiez le nom de colonne.")

# --- Créer colonnes date-only (yyyy-mm-dd) pour l'alignement ---
df_a['__date_only'] = df_a[date_col_a].dt.date  # peut contenir NaT -> donne NaN (None) si NaT
df_b['__date_only'] = pd.to_datetime(df_b[date_col_b]).dt.date

# --- Préparer mapping date -> range_m (si plusieurs en B par date, on prend le premier) ---
df_b_map = df_b.dropna(subset=['__date_only']).drop_duplicates(subset=['__date_only']).set_index('__date_only')['range_m']

# --- Joindre range_m aux lignes existantes de A (même date pour toutes les lignes du jour) ---
df_a = df_a.copy()
df_a['range_m'] = df_a['__date_only'].map(df_b_map)

# --- Trouver les dates qui sont dans B mais pas dans A et créer lignes vides ---
dates_in_b = set(df_b_map.index)
dates_in_a = set(df_a['__date_only'].dropna().unique())
missing_dates = sorted(dates_in_b - dates_in_a)

if missing_dates:
    # colonnes à répliquer pour les nouvelles lignes (toutes NaN sauf date et range_m)
    cols = df_a.columns.tolist()
    # on enlève la colonne auxiliaire '__date_only' de la liste pour gérer séparément
    if '__date_only' in cols:
        cols.remove('__date_only')

    new_rows = []
    for d in missing_dates:
        row = {c: np.nan for c in cols}
        # mettre la colonne date au format ISO avec Z (pour ressembler à A)
        # si A a au moins une valeur de timezone utc on formatera en 'YYYY-MM-DDT00:00:00Z'
        ts_iso_z = pd.Timestamp(d).strftime('%Y-%m-%dT%H:%M:%SZ')  # 00:00:00Z
        row[date_col_a] = ts_iso_z
        row['range_m'] = df_b_map.loc[d]
        new_rows.append(row)

    df_new = pd.DataFrame(new_rows, columns=cols)
    # pour la cohérence, parser la colonne date nouvellement créée en datetime (utc)
    df_new[date_col_a] = pd.to_datetime(df_new[date_col_a], utc=True, errors='coerce')
    # ajouter colonne '__date_only' aux nouvelles lignes
    df_new['__date_only'] = df_new[date_col_a].dt.date

    # concaténer
    df_result = pd.concat([df_a, df_new], ignore_index=True, sort=False)
else:
    df_result = df_a

# --- Optionnel : trier par date si vous voulez ---
# si la colonne date est datetime (ce qu'on a essayé), on peut trier
try:
    df_result = df_result.sort_values(by=date_col_a).reset_index(drop=True)
except Exception:
    pass

# --- Nettoyage : garder colonnes originales + range_m (et supprimer la colonne auxiliaire) ---
if '__date_only' in df_result.columns:
    df_result = df_result.drop(columns='__date_only')

# écrire en CSV
df_result.to_csv(output_csv, index=False)

print(f"Terminé. Fichier de sortie créé : {output_csv}")
print(f"Lignes totales : {len(df_result)}. Dates ajoutées : {len(missing_dates)}.")
