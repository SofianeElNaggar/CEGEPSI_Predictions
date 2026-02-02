import pandas as pd

# Nom du fichier
input_csv = "../../dataset/OMDS-CTD-meteogc-data.csv"
output_csv = "../../dataset/OMDS-CTD-meteogc-data.csv"

# Charger seulement la date comme string pour éviter un parse lent
df = pd.read_csv(input_csv)

# Conversion datetime (optimisée)
df['time (UTC)'] = pd.to_datetime(df['time (UTC)'], utc=True)

# Pour chaque date identique, numéroter les doublons : 0,1,2,...
df['offset'] = df.groupby('time (UTC)').cumcount()

# Ajouter les secondes correspondantes
df['time (UTC)'] = df['time (UTC)'] + pd.to_timedelta(df['offset'], unit='s')

# Supprimer la colonne temporaire
df = df.drop(columns=['offset'])

# Export propre
df.to_csv(output_csv, index=False)
