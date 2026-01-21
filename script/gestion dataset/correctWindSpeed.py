import pandas as pd
import sys

def remplacer_inf_31_csv(fichier_entree, fichier_sortie):
    # Lecture du CSV
    df = pd.read_csv(fichier_entree)

    # Remplacement de "<31" par 15
    df = df.replace("<31", 15)

    # Conversion des colonnes en numérique quand possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Sauvegarde en CSV
    df.to_csv(fichier_sortie, index=False)

if __name__ == "__main__":
    fichier_entree = '../../dataset/OMDS-CTD-meteogc-data.csv'
    fichier_sortie = '../../dataset/OMDS-CTD-meteogc-data2.csv'

    remplacer_inf_31_csv(fichier_entree, fichier_sortie)
    print("Traitement terminé.")
