import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pyarrow.parquet as pq
import os

# ------------ Paramètres à modifier ------------
INPUT_FILE = "../../dataset/OMDS-CTD-meteogc-data.parquet"       # Nom du fichier parquet à lire
DATE_COLUMN = "time (UTC)"              # Nom de la colonne date
OUTPUT_PDF = "plots_2020_2025.pdf"
# ------------------------------------------------

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Fichier introuvable : {INPUT_FILE}")
        return

    # Lecture parquet
    df = pd.read_parquet(INPUT_FILE)

    # Conversion en datetime (gère automatiquement les timezone)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], utc=True)

    print(df[DATE_COLUMN].head())
    print(df[DATE_COLUMN].dtype)

    # Filtrage des dates
    df_filtered = df[
        (df[DATE_COLUMN].dt.year >= 2020) &
        (df[DATE_COLUMN].dt.year <= 2025)
    ]

    # Colonnes numériques
    numeric_cols = df_filtered.select_dtypes(include='number').columns

    if len(numeric_cols) == 0:
        print("Aucune colonne numérique à tracer.")
        return

    # Génération PDF
    with PdfPages(OUTPUT_PDF) as pdf:
        for col in numeric_cols:
            plt.figure(figsize=(10, 4))
            plt.plot(df_filtered[DATE_COLUMN], df_filtered[col])
            plt.title(f"{col} (2020–2025)")
            plt.xlabel("Date")
            plt.ylabel(col)
            plt.grid(True)

            pdf.savefig()
            plt.close()

    print(f"PDF généré : {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
