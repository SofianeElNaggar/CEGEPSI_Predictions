# utils.py
# Constantes partagées et utilitaires légers

PARQUET_PATH = "dataset/OMDS-CTD-meteogc-data.parquet"

# profondeur ciblée (ex 1.0m +/- tol)
DEPTH_CENTER = 1.0
DEPTH_TOLERANCE = 0.1

# Agg: 'median' ou 'mean'
AGG_METHOD = "mean"

# template pour les PDFs de sortie (on injectera le nom de la combinaison)
OUTPUT_PDF_TEMPLATE = f"results/prediction/LSTM/PINN/{AGG_METHOD}/LSTM_predictions_{AGG_METHOD}.pdf"

# Période d'utilisation des données
START_DATE = "2000-01-01"
TRAIN_END = "2020-01-01"
TEST_END = "2025-01-01"

# LSTM / entrainement
SEQUENCE_LENGTH = 60   
N_EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 8

# Fraction du jeu d'entraînement à réserver comme validation
VALIDATION_FRAC = 0.05

RECURSIVE_FORECAST = True

ALL_TARGETS = [
    #"temperature (°C)",
    #"chlorophyll (mg m-3)",
    #"turbidity (NTU)",
    "pH",
    #"salinity (PSS-78)",
    #"dissolved_oxygen (ml l-1)",
]

# Colonnes utilisées UNIQUEMENT comme entrées (jamais prédites)
INPUT_ONLY_COLS = [
    "temperature (°C)",
    "chlorophyll (mg m-3)",
    #"turbidity (NTU)",
    #"pH",
    "salinity (PSS-78)",
    #"potential_density (kg m-3)",
    "dissolved_oxygen (ml l-1)",
    #"pressure (dbar)",

    #------------------------------------

    "tide_range (m)",
    #"Max Temp (°C)",
    #"Min Temp (°C)",
    "Mean Temp (°C)",
    #"Heat Deg Days (°C)",
    #"Cool Deg Days (°C)",
    #"Total Precip (mm)",
    #"Dir of Max Gust (10s deg)",
    "Spd of Max Gust (km/h)"

]

# Features temporelles
TIME_FEATURE_COLS = ["doy_sin", "doy_cos"]