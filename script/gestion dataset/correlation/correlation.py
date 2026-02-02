# -*- coding: utf-8 -*-
"""
correlation_targets_only_coolwarm.py

Calcule la matrice de corrélation sur toutes les données (toutes profondeurs, toutes dates),
uniquement pour les colonnes dans TARGET_COLS.
Applique le filtrage QC (valeurs 1, 5 ou NaN).
Sauvegarde la heatmap en PDF.

Auteur : ChatGPT (GPT-5)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np

# === PARAMÈTRES ===
PARQUET_PATH = "../../dataset/OMDS-CTD datalight_with_pos.parquet"
OUTPUT_PDF = "../results/correlation/correlation_matrix.pdf"

TARGET_COLS = [
    "depth", "longitude", "latitude",
    "TEMPS901", "CPHLPR01", "TURBPR01",
    "PHXXPR01", "PSALST01", "SIGTEQST", "DOXYZZ01"
]


def load_and_filter_qc(parquet_path):
    print("Chargement du fichier :", parquet_path)
    df = pd.read_parquet(parquet_path)

    # Conversion des types
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
    if 'depth' in df.columns:
        df['depth'] = pd.to_numeric(df['depth'], errors='coerce')

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filtrage QC
    qc_cols = [c for c in df.columns if c.upper().endswith('_QC')]
    print("Colonnes QC détectées :", qc_cols)
    mask = pd.Series(True, index=df.index)
    for qc in qc_cols:
        mask &= (df[qc].isna() | (df[qc] == 1) | (df[qc] == 5))
    df = df[mask]
    print(f"Lignes restantes après filtrage QC : {len(df)}")

    return df


def plot_correlation(df, target_cols, output_pdf):
    """Calcule et trace la matrice de corrélation"""
    cols = [c for c in target_cols if c in df.columns]
    corr = df[cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Matrice de corrélation (toutes profondeurs, toutes dates)")
    plt.tight_layout()

    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        pdf.savefig(bbox_inches="tight")
    print(f"PDF sauvegardé : {out}")


def main():
    df = load_and_filter_qc(PARQUET_PATH)
    plot_correlation(df, TARGET_COLS, OUTPUT_PDF)


if __name__ == "__main__":
    main()
