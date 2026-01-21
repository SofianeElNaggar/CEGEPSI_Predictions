# -*- coding: utf-8 -*-
"""
correlation_matrix_csv_adapted.py

Calcule la matrice de corrélation à partir d'un CSV comme celui fourni
(la 1ère ligne est l'en-tête et la 2ème ligne contient les unités).

Fonctionnalités :
- lit un CSV en ignorant la ligne d'unités (ligne 2)
- convertit automatiquement les colonnes en numériques (en nettoyant les valeurs
  comme "<31" -> 31)
- applique le filtrage QC si des colonnes se terminent par "_QC" (acceptées : NaN, 1, 5)
- calcule la matrice de corrélation sur un sous-ensemble de colonnes cibles
- sauvegarde une heatmap (palette coolwarm) en PDF

Usage :
    python correlation_matrix_csv_adapted.py --csv path/to/file.csv --out results/corr.pdf

Auteur : ChatGPT (GPT-5 Thinking mini)
"""

from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# === PARAMÈTRES PAR DÉFAUT ===
DEFAULT_CSV = "../../dataset/OMDS-CTD-meteogc-data.csv"
DEFAULT_OUTPUT = "../../results/correlation/correlation_matrix_OMDS-meteogc.pdf"

# Colonnes cibles (nom exact tel qu'il apparaît dans l'en-tête CSV)
# Vous pouvez modifier cette liste selon les colonnes qui vous intéressent.
TARGET_COLS = [
    "time", "latitude", "longitude", "depth", "pressure",
    "par_downwelling", "par_upwelling", "temperature", "chlorophyll",
    "turbidity", "pH", "salinity", "potential_density", "dissolved_oxygen",
    "tide_range", "Max Temp", "Min Temp", "Mean Temp",
    "Heat Deg Days", "Cool Deg Days", "Total Rain", "Total Snow",
    "Total Precip", "Snow on Grnd", "Dir of Max Gust", "Spd of Max Gust"
]

# --- Fonctions utilitaires ---

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Tente de convertir une série en numérique.

    - Supprime les caractères non numériques courants (par ex. '<', '>' et espaces)
    - Remplace les valeurs vides / impossibles par NaN
    - Conserve les types numériques existants
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    def to_num(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        # chaîne : retirer espaces, caractères non désirés
        x = str(x).strip()
        if x == "":
            return np.nan
        # enlever les annotations comme '<31' ou '>5' -> on retire le signe et on parse
        x = re.sub(r"^[<>]=?", "", x)
        # enlever tout sauf chiffres, point, signe moins et exponentielle
        x = re.sub(r"[^0-9eE+\-\.]+", "", x)
        try:
            return float(x)
        except Exception:
            return np.nan

    return s.map(to_num).astype(float)


def load_csv_skip_units(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV en sautant la ligne d'unités (ligne 2) si présente.

    On suppose que la première ligne contient les noms de colonnes.
    Si le CSV n'a pas de ligne d'unités, skiprows=[1] n'est pas dommageable si
    cette ligne n'existe pas (pandas lève une erreur). Pour être sûr, on ouvre
    le fichier et regarde la deuxième ligne : si elle contient des caractères
    non-alphanumériques majoritairement (ex: 'UTC,degrees_north,...'), on la saute.
    """
    # Lire les deux premières lignes pour décider
    with open(csv_path, 'r', encoding='utf-8') as f:
        first = f.readline()
        second = f.readline()

    skip_second = False
    if second is not None:
        # heuristique simple : si deuxième ligne contient '/' ou 'degrees' ou 'm' ou 'UTC' ou 'MicroEinsteins' etc.
        # on considère qu'il s'agit d'une ligne d'unités/description et on la saute
        if re.search(r"degrees|UTC|MicroEinsteins|m\-|m\^|m,|m |dbar|NTU|PSS|mg|ml|kg|°C|mm|cm", second, flags=re.IGNORECASE):
            skip_second = True

    if skip_second:
        df = pd.read_csv(csv_path, header=0, skiprows=[1], low_memory=False)
    else:
        df = pd.read_csv(csv_path, header=0, low_memory=False)

    return df


def apply_qc_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le filtrage QC si des colonnes se terminent par _QC.

    Critère : conserver les lignes où chaque QC est NaN ou égal à 1 ou 5.
    Si aucune colonne QC n'est détectée, on ne filtre pas.
    """
    qc_cols = [c for c in df.columns if str(c).upper().endswith('_QC')]
    if not qc_cols:
        print("Aucune colonne QC détectée. Aucun filtrage QC appliqué.")
        return df

    mask = pd.Series(True, index=df.index)
    for qc in qc_cols:
        mask &= (df[qc].isna() | (df[qc] == 1) | (df[qc] == 5))
    df_filtered = df[mask].copy()
    print(f"Filtrage QC appliqué : {len(df)} -> {len(df_filtered)} lignes")
    return df_filtered


def plot_and_save_correlation(df: pd.DataFrame, target_cols: list, output_pdf: Path):
    """Calcule la corrélation sur les colonnes numériques cibles et sauvegarde la heatmap."""
    # garder seulement les colonnes demandées qui existent
    cols_present = [c for c in target_cols if c in df.columns]
    if not cols_present:
        raise ValueError("Aucune des colonnes cibles n'est présente dans le CSV.")

    # créer une copie nettoyée : convertir chaque colonne non numérique
    cleaned = pd.DataFrame(index=df.index)
    for c in cols_present:
        if c.lower() == 'time' and c in df.columns:
            # parser la colonne time
            cleaned['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
            # pour la corrélation, on peut convertir en timestamp (float) si on veut utiliser le temps
            cleaned['time'] = cleaned['time'].astype('int64') / 1e9  # secondes depuis epoch
        else:
            cleaned[c] = clean_numeric_series(df[c])

    # sélectionner uniquement les colonnes numériques (après nettoyage)
    numeric_cols = [c for c in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[c])]
    if not numeric_cols:
        raise ValueError("Aucune colonne numérique disponible pour calculer la corrélation.")

    corr = cleaned[numeric_cols].corr()

    plt.figure(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Matrice de corrélation")
    plt.tight_layout()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(bbox_inches='tight')
    plt.close()
    print(f"Heatmap sauvegardée : {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Calculer une matrice de corrélation depuis un CSV avec ligne d'unités.")
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV, help='Chemin vers le fichier CSV')
    parser.add_argument('--out', type=str, default=DEFAULT_OUTPUT, help='Chemin du PDF de sortie')
    parser.add_argument('--cols', type=str, nargs='*', default=None,
                        help='Liste optionnelle de colonnes cibles (si non fourni, TARGET_COLS est utilisé)')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_pdf = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    df = load_csv_skip_units(csv_path)

    # Option : remplacer la liste TARGET_COLS par celle fournie
    cols_to_use = args.cols if args.cols else TARGET_COLS

    # Appliquer filtrage QC si présent
    df = apply_qc_filter(df)

    # Calculer et sauvegarder
    plot_and_save_correlation(df, cols_to_use, out_pdf)


if __name__ == '__main__':
    main()
