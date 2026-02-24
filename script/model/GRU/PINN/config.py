# config.py
import os

class Config:
    # --- Chemins ---
    PARQUET_PATH = "dataset/OMDS-CTD-meteogc-data.parquet"

    # --- Prétraitement ---
    DEPTH_CENTER    = 1.0    # Profondeur cible (m)
    DEPTH_TOLERANCE = 0.1    # Tolérance autour de la profondeur cible
    AGG_METHOD      = "mean" # Méthode d'agrégation journalière : 'mean' ou 'median'

    # --- Période temporelle ---
    START_DATE = "2000-01-01"
    TRAIN_END  = "2020-01-01"
    TEST_END   = "2025-01-01"

    # --- Hyperparamètres d'entraînement ---
    SEQUENCE_LENGTH = 60
    N_EPOCHS        = 60
    BATCH_SIZE      = 64
    LEARNING_RATE   = 1e-3
    PATIENCE        = 8
    VALIDATION_FRAC = 0.15

    # --- Mode de prédiction ---
    RECURSIVE_FORECAST = True  # True : récursif (pas d'observations futures), False : walk-forward

    # ── Architecture du modèle ────────────────────────────────────────────────
    RNN_TYPE = "LSTM"   # Cellule récurrente : "GRU" | "LSTM"
    USE_CNN  = True    # True : CNN devant le RNN, False : RNN seul

    # Hyperparamètres du modèle
    HIDDEN_SIZE      = 128
    HIDDEN_SIZE_2    = 64
    DROPOUT          = 0.2
    CNN_OUT_CHANNELS = 64   # Nombre de filtres CNN (ignoré si USE_CNN = False)

    # ── Hyperparamètres VMD ────────────────────────────────────────────────────
    VMD_ALPHA = 2000       # Contrainte de bande passante
    VMD_TAU   = 0.01       # Tolérance au bruit
    VMD_K     = 8          # Nombre de modes
    VMD_DC    = 0          # Pas de composante DC imposée
    VMD_INIT  = 0          # Initialisation uniforme des fréquences centrales
    VMD_TOL   = 1e-7       # Tolérance de convergence

    # ── Hyperparamètres CEEMDAN ───────────────────────────────────────────────
    CEEMDAN_TRIALS   = 100   # Nombre de réalisations d'ensemble
    CEEMDAN_EPSILON  = 0.2   # Amplitude du bruit ajouté (fraction de l'écart-type)
    CEEMDAN_MAX_IMFS = None  # Nombre max d'IMFs à conserver (None = toutes)

    # ── Hyperparamètres SSA ───────────────────────────────────────────────────
    SSA_WINDOW = 365         # Taille de la fenêtre de Hankel (en jours)

    # --- Poids de la fonction de perte ---
    RNN_LOSS_WEIGHT  = 3.0   # Poids de la perte MSE du RNN
    PINN_LOSS_WEIGHT = 1.0   # Multiplicateur global de la perte PINN

    # Poids individuels par contrainte physique
    PINN_WEIGHTS = {
        'doy':              0.0,
        'dissolved_oxygen': 1.0,
        'ph':               1.0
    }

    # --- Variables cibles et d'entrée ---
    ALL_TARGETS = [
        "temperature (°C)",
        #"chlorophyll (mg m-3)",
        #"turbidity (NTU)",
        #"pH",
        #"salinity (PSS-78)",
        "dissolved_oxygen (ml l-1)",
    ]

    # Colonnes utilisées uniquement en entrée (jamais prédites)
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

    TIME_FEATURE_COLS = ["doy_sin", "doy_cos"]

    # ── Méthode de décomposition ──────────────────────────────────────────────
    # Choisir : "VMD" | "CEEMDAN" | "SSA" | False (aucune décomposition)
    DECOMPOSITION_METHOD = "SSA"

    # Colonnes sur lesquelles appliquer la décomposition
    DECOMPOSITION_COLS = ALL_TARGETS  # + INPUT_ONLY_COLS

    # --- Sortie ---
    OUTPUT_DIR          = f"results/prediction/v1/{AGG_METHOD}"
    OUTPUT_PDF_TEMPLATE = f"{OUTPUT_DIR}/v1_predictions_{AGG_METHOD}.pdf"

    @classmethod
    def get_output_path(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls.OUTPUT_PDF_TEMPLATE
