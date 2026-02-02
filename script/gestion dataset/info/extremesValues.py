# extremesValues_fixed.py
import pandas as pd
import numpy as np

# ---- paramètres ----
CSV_PATH = r"../../dataset/OMDS-CTD-meteogc-data.csv"
COLUMN = "depth"   # <-- remplace par le nom réel de ta colonne
OUTLIERS_CSV = r"outliers.csv"  # <-- chemin où sauvegarder (optionnel)
# --------------------

def try_numeric_conversion(col_series: pd.Series) -> pd.Series:
    """
    Essaie plusieurs stratégies pour convertir en numérique :
    1) to_numeric direct (errors='coerce')
    2) nettoyage : suppression de caractères non numériques sauf '-' et '.' et remplacement ',' -> '.'
    3) fusion des deux résultats (préférer le 1) et retour de la série numérique (float) avec NaN si impossible.
    """
    s = col_series.copy().astype(str).fillna("").str.strip()

    # 1) conversion directe (utile si la colonne est déjà principalement numeric)
    num_direct = pd.to_numeric(s, errors="coerce")

    # 2) nettoyage plus agressif
    # - garder chiffres, signe moins, virgule, point et éventuellement exposant E,e
    cleaned = s.str.replace(r"(?i)[^0-9eE\+\-\,\.\s]", "", regex=True)   # supprime symboles bizarres
    cleaned = cleaned.str.replace(r"\s+", "", regex=True)               # supprime espaces
    cleaned = cleaned.str.replace(",", ".", regex=False)                # virgule -> point
    num_cleaned = pd.to_numeric(cleaned, errors="coerce")

    # 3) composer : privilégier num_direct, sinon num_cleaned
    composed = num_direct.copy()
    composed[pd.isna(composed)] = num_cleaned[pd.isna(composed)]
    return composed

def main():
    df = pd.read_csv(CSV_PATH, dtype=str)  # on lit en str pour inspecter/convertir proprement
    if COLUMN not in df.columns:
        raise ValueError(f"La colonne '{COLUMN}' n'existe pas dans le fichier. Colonnes disponibles: {list(df.columns)}")

    # Série d'origine (strings)
    original = df[COLUMN]

    # Conversion numérique robuste
    numeric = try_numeric_conversion(original)

    # Lignes non converties
    non_numeric_mask = numeric.isna() & original.notna() & (original.astype(str) != "")
    non_numeric_count = non_numeric_mask.sum()

    if non_numeric_count > 0:
        print(f"Attention : {non_numeric_count} valeur(s) n'ont pas pu être converties en nombre.")
        print("Exemples (premières 20 lignes non-convertibles) :")
        print(df.loc[non_numeric_mask, [COLUMN]].head(20).to_string(index=True))
        print("\nSi ce sont des unités (e.g. '12 m'), ou du texte, nettoie la colonne ou adapte le script.\n")

    # Série propre pour calculs
    clean_vals = numeric.dropna()
    if clean_vals.empty:
        raise ValueError("Aucune valeur numérique valide trouvée dans la colonne après conversion.")

    # Calcul percentiles 1% et 99%
    p99 = clean_vals.quantile(0.999)

    print(f"99e percentile (99%) : {p99}\n")

    # Lignes en dehors de [p1, p99]
    outliers_df = df.loc[mask_outliers].copy()
    # ajouter colonne numérique convertie pour info
    outliers_df["_numeric_value"] = numeric[mask_outliers]

    if outliers_df.empty:
        print("Aucun outlier selon l'intervalle [1%, 99%].")
    else:
        print(f"{len(outliers_df)} ligne(s) en dehors de l'intervalle [p1, p99] :")
        # affiche quelques colonnes : index + la colonne + valeur numérique
        print(outliers_df[[COLUMN, "_numeric_value"]].to_string(index=True))

        # Option : sauvegarder dans un CSV
        try:
            outliers_df.to_csv(OUTLIERS_CSV, index=True)
            print(f"\nLes outliers ont été sauvegardés dans : {OUTLIERS_CSV}")
        except Exception as e:
            print(f"\nImpossible de sauvegarder le CSV d'outliers : {e}")

if __name__ == "__main__":
    main()
