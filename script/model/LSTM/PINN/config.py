# config.py
import os

class Config:
    # --- Data Paths ---
    PARQUET_PATH = "dataset/OMDS-CTD-meteogc-data.parquet"
    
    # --- Preprocessing ---
    # profondeur ciblée (ex 1.0m +/- tol)
    DEPTH_CENTER = 1.0
    DEPTH_TOLERANCE = 0.1
    # Agg: 'median' ou 'mean'
    AGG_METHOD = "mean"
    
    # Période d'utilisation des données
    START_DATE = "2000-01-01"
    TRAIN_END = "2020-01-01"
    TEST_END = "2025-01-01"

    # --- Training Hyperparameters ---
    SEQUENCE_LENGTH = 60   
    N_EPOCHS = 60
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    PATIENCE = 8
    VALIDATION_FRAC = 0.05
    
    # --- Prediction ---
    RECURSIVE_FORECAST = True
    
    # --- Weights (Loss) ---
    # Global weights
    LSTM_LOSS_WEIGHT = 1.0
    PINN_LOSS_WEIGHT = 1.0  # Multiplicateur global pour la partie PINN

    # Individual PINN weights
    PINN_WEIGHTS = {
        'doy': 1.0,
        'dissolved_oxygen': 1.0,
        'ph': 1.0
    }

    # --- Target & Feature Definitions ---
    ALL_TARGETS = [
        "temperature (°C)",
        #"chlorophyll (mg m-3)",
        #"turbidity (NTU)",
        #"pH",
        #"salinity (PSS-78)",
        "dissolved_oxygen (ml l-1)",
    ]
    
    # Colonnes utilisées UNIQUEMENT comme entrées (jamais prédites)
    INPUT_ONLY_COLS = [
        "temperature (°C)",
        "chlorophyll (mg m-3)",
        #"turbidity (NTU)",
        #"pH",
        "salinity (PSS-78)",
        #"potential_density (kg m-3)",
        #"dissolved_oxygen (ml l-1)",
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
    
    TIME_FEATURE_COLS = ["doy_sin", "doy_cos"]

    # --- Output ---
    OUTPUT_DIR = f"results/prediction/LSTM/PINN/{AGG_METHOD}"
    OUTPUT_PDF_TEMPLATE = f"{OUTPUT_DIR}/LSTM_predictions_{AGG_METHOD}.pdf"
    
    @classmethod
    def get_output_path(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls.OUTPUT_PDF_TEMPLATE
