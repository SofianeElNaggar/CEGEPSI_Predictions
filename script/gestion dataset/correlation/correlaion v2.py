# -*- coding: utf-8 -*-
"""
Script : correlation_matrix.py
But : Calculer la matrice de corrélation du dataset .parquet et l'enregistrer en PDF.
Remarques :
 - Filtre QC (conserve lignes où chaque *_QC est NaN ou 1 ou 5)
 - Optionnel : filtrage par profondeur (centre +/- tol)
 - Agrégation journalière (médiane ou moyenne) avant corrélation (configurable)
 - Trace une heatmap avec annotations et sauvegarde en PDF
"""
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ------------------ RÉGLAGES UTILISATEUR ------------------
PARQUET_PATH = "../../dataset/OMDS-CTD datalight_with_pos.parquet"  # chemin vers ton .parquet


# Filtrage / prétraitement
START_DATE = "2000-01-01"         # ignorer les données avant (mettre None pour ne pas filtrer)
APPLY_DEPTH_FILTER = True         # True pour filtrer par profondeur
DEPTH_CENTER = 15.0                # profondeur cible
DEPTH_TOLERANCE = 0.2             # tolérance +/- (m)

# Agrégation journalière (recommandé si plusieurs mesures par jour/profondeur)
AGG_METHOD = "median"             # 'median' ou 'mean' ou None (None -> pas d'agrégation)


OUTPUT_PDF = f"../../results/correlation/correlation_matrix_depth={str(DEPTH_CENTER)}_method={AGG_METHOD}.pdf"      # fichier PDF de sortie

# Colonnes cibles à inclure par défaut dans la corrélation (tu peux ajouter/supprimer)
TARGET_COLS = [
    "TEMPS901",
    "CPHLPR01",
    "TURBPR01",
    "PHXXPR01",
    "PSALST01",
    "SIGTEQST",
    "DOXYZZ01",
]

# Si tu veux inclure *toutes* les colonnes numériques trouvées dans le dataset, mets INCLUDE_ALL_NUMERIC=True.
INCLUDE_ALL_NUMERIC = False
# ---------------------------------------------------------

def ensure_output_path(path_str):
    p = Path(path_str)
    if p.suffix == "":
        p = p.with_suffix(".pdf")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print("Impossible de créer dossier sortie :", e)
    return p

def load_and_preprocess(parquet_path, start_date=None, depth_center=None, depth_tol=None, apply_depth=False):
    print("Chargement :", parquet_path)
    df = pd.read_parquet(parquet_path)

    # --- Time to tz-aware UTC if possible ---
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)

    # Convertir depth en numeric si present
    if 'depth' in df.columns:
        df['depth'] = pd.to_numeric(df['depth'], errors='coerce')

    # Convertir colonnes QC et colonnes cibles en numeric (sécurité)
    qc_cols = [c for c in df.columns if c.upper().endswith('_QC')]
    for c in qc_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in TARGET_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Filtrer start_date si demandé
    if start_date is not None:
        start_ts = pd.to_datetime(start_date, errors='coerce')
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize('UTC')
        else:
            start_ts = start_ts.tz_convert('UTC')
        if 'time' not in df.columns:
            raise RuntimeError("La colonne 'time' est absente du dataset; impossible de filtrer par date.")
        df = df[df['time'] >= start_ts]

    # Appliquer filtrage QC : conserver lignes où chaque QC est NaN ou 1 ou 5
    if qc_cols:
        mask_good = pd.Series(True, index=df.index)
        for qc in qc_cols:
            mask_good &= ((df[qc].isna()) | (df[qc] == 1) | (df[qc] == 5))
        df = df[mask_good]
        print(f"Après filtrage QC : {len(df)} lignes restantes (qc cols: {qc_cols})")
    else:
        print("Aucune colonne *_QC détectée -> pas de filtrage QC appliqué.")

    # Filtre profondeur optionnel
    if apply_depth and 'depth' in df.columns and depth_center is not None and depth_tol is not None:
        mask_depth = df['depth'].notna() & (np.abs(df['depth'] - depth_center) <= depth_tol)
        df = df[mask_depth]
        print(f"Après filtrage profondeur (center={depth_center}, tol={depth_tol}) : {len(df)} lignes restantes")
    elif apply_depth:
        print("apply_depth=True mais 'depth' absent ou paramètres manquants -> pas de filtrage profondeur.")

    return df

def aggregate_daily(df, cols, method='median'):
    # si pas de colonne 'time', on suppose df index déjà en dates
    if 'time' in df.columns:
        df2 = df.copy()
        df2['date'] = df2['time'].dt.floor('D')
    else:
        df2 = df.copy()
        if df2.index.dtype == 'datetime64[ns]':
            df2['date'] = df2.index.floor('D')
        else:
            raise RuntimeError("Aucune colonne 'time' et index non datetime -> impossible d'agréger par jour.")

    if method == 'median':
        agg = df2.groupby('date')[cols].median()
    elif method == 'mean':
        agg = df2.groupby('date')[cols].mean()
    else:
        raise ValueError("method doit être 'median' ou 'mean'")

    agg.index = pd.to_datetime(agg.index)
    return agg

def make_correlation_matrix(df, include_all_numeric=False, target_cols=None, agg_method='median'):
    # Sélection des colonnes à utiliser
    if include_all_numeric:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'time' in numeric:
            numeric.remove('time')
        cols_use = numeric
    else:
        # ne garder que les colonnes target existantes dans df
        cols_use = [c for c in (target_cols or []) if c in df.columns]

    if len(cols_use) == 0:
        raise RuntimeError("Aucune colonne sélectionnée pour la matrice de corrélation.")

    print("Colonnes utilisées pour corrélation :", cols_use)

    # Option : agrégation journalière si le df a une colonne 'time' et qu'on a plusieurs mesures/jour
    if agg_method is not None and 'time' in df.columns:
        df_daily = aggregate_daily(df, cols_use, method=agg_method)
    else:
        # si pas d'agrégation, on prend les colonnes sélectionnées et dropna pour corr
        df_daily = df[cols_use].copy()
        if df_daily.index.dtype != 'datetime64[ns]' and 'time' in df.columns:
            df_daily.index = pd.to_datetime(df['time']).floor('D')

    # Drop columns fully NA
    df_daily = df_daily.dropna(axis=1, how='all')
    # Optionnel : on peut aussi drop rows with all NA
    df_daily = df_daily.dropna(axis=0, how='all')

    # Calculer corr (pearson)
    corr = df_daily.corr(method='pearson')
    return corr, df_daily

def plot_and_save_correlation(corr, out_pdf_path, title="Matrice de corrélation", figsize=(12,10), vmax=None):
    p = Path(out_pdf_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Figure
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.0)
    # Palette demandée : 'coolwarm' (contraste chaud/froid)
    cmap = 'coolwarm'
    # draw heatmap — forcer l'échelle sur [-1, 1] pour les corrélations
    ax = sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        vmin=-1 if vmax is None else -vmax,
        vmax=1 if vmax is None else vmax,
        cbar_kws={"shrink": .75},
        linewidths=0.5
    )
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save single-page PDF
    try:
        with PdfPages(out_pdf_path) as pdf:
            pdf.savefig(bbox_inches='tight')
        print("PDF sauvegardé :", out_pdf_path)
    except Exception as e:
        print("Erreur lors de la sauvegarde du PDF :", e)

# ------------------ Exécution ------------------
def main():
    out_path = ensure_output_path(OUTPUT_PDF)
    df = load_and_preprocess(PARQUET_PATH, START_DATE, DEPTH_CENTER, DEPTH_TOLERANCE, APPLY_DEPTH_FILTER)

    # Choix des colonnes: par défaut TARGET_COLS, ou toutes les numériques
    corr, df_used = make_correlation_matrix(df, include_all_numeric=INCLUDE_ALL_NUMERIC, target_cols=TARGET_COLS, agg_method=AGG_METHOD)

    print("Matrice de corrélation calculée. Dimensions :", corr.shape)
    # Afficher un petit aperçu
    print(corr.round(2))

    # Plot & save
    title = f"Matrice de corrélation ({AGG_METHOD}) — colonnes: {', '.join(corr.columns)}"
    plot_and_save_correlation(corr, out_path, title=title)

if __name__ == "__main__":
    main()
