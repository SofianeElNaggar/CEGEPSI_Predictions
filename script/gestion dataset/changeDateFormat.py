import pandas as pd

# Chemins des fichiers
input_csv = "../../dataset/OMDS-CTD-meteogc-data2.csv"
output_csv = "../../dataset/OMDS-CTD-meteogc-data-v2.csv"

# Lecture du CSV
df = pd.read_csv(input_csv)

# Conversion du format de date
df["time (UTC)"] = (
    pd.to_datetime(df["time (UTC)"], utc=True)
    .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
)

# Sauvegarde du nouveau CSV
df.to_csv(output_csv, index=False)

print("Conversion terminée !")
