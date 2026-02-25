# main.py
import traceback
import torch
from config import Config
from data_manager import DataManager
from model import CNNGRUModel, CNNLSTMModel, GRUModel, LSTMModel
from trainer import Trainer
from evaluator import Evaluator
from pinns import CosSinPINN, DissolvedOxygenPINN, pHPINN

def main():
    try:
        # Chargement de la configuration
        config = Config()

        # Préparation des données (décomposition, normalisation, séquences)
        dm = DataManager(config)
        dm.prepare_data(target_cols=config.ALL_TARGETS)

        # Instanciation du modèle selon config.RNN_TYPE et config.USE_CNN
        if config.USE_CNN and config.RNN_TYPE == "GRU":
            model = CNNGRUModel(
                n_features=dm.n_features,
                n_outputs=dm.n_outputs,
                hidden_size=config.HIDDEN_SIZE,
                hidden2=config.HIDDEN_SIZE_2,
                dropout=config.DROPOUT,
                cnn_out_channels=config.CNN_OUT_CHANNELS
            )
        elif config.USE_CNN and config.RNN_TYPE == "LSTM":
            model = CNNLSTMModel(
                n_features=dm.n_features,
                n_outputs=dm.n_outputs,
                hidden_size=config.HIDDEN_SIZE,
                hidden2=config.HIDDEN_SIZE_2,
                dropout=config.DROPOUT,
                cnn_out_channels=config.CNN_OUT_CHANNELS
            )
        elif config.RNN_TYPE == "LSTM":
            model = LSTMModel(
                n_features=dm.n_features,
                n_outputs=dm.n_outputs,
                hidden_size=config.HIDDEN_SIZE,
                hidden2=config.HIDDEN_SIZE_2,
                dropout=config.DROPOUT
            )
        else:  # GRU par défaut
            model = GRUModel(
                n_features=dm.n_features,
                n_outputs=dm.n_outputs,
                hidden_size=config.HIDDEN_SIZE,
                hidden2=config.HIDDEN_SIZE_2,
                dropout=config.DROPOUT
            )
        print(f"Modèle : {'CNN-' if config.USE_CNN else ''}{config.RNN_TYPE}")

        # Définition des contraintes physiques (PINNs)
        pinns = [
            CosSinPINN(
                'doy_sin',
                'doy_cos',
                in_targets=False,
                weight=config.PINN_WEIGHTS.get('doy')
            ),
        ]

        if "dissolved_oxygen (ml l-1)" in dm.target_cols:
            pinns.append(
                DissolvedOxygenPINN(
                    do_name="dissolved_oxygen (ml l-1)",
                    temp_water_name="temperature (°C)",
                    temp_air_name="Mean Temp (°C)",
                    chl_name="chlorophyll (mg m-3)",
                    wind_name="Spd of Max Gust (km/h)",
                    sal_name="salinity (PSS-78)",
                    tide_name="tide_range (m)",
                    weight=config.PINN_WEIGHTS.get('dissolved_oxygen')
                )
            )

        if "pHa" in dm.target_cols:
            pinns.append(
                pHPINN(
                    ph_name="pH",
                    temp_water_name="temperature (°C)",
                    sal_name="salinity (PSS-78)",
                    chl_name="chlorophyll (mg m-3)",
                    do_name="dissolved_oxygen (ml l-1)",
                    wind_name="Spd of Max Gust (km/h)",
                    tide_name="tide_range (m)",
                    weight=config.PINN_WEIGHTS.get('ph')
                )
            )

        # Entraînement
        trainer = Trainer(model, config, pinns=pinns)
        trainer.train(
            dm.train_loader,
            dm.val_loader,
            feature_cols=dm.feature_cols,
            target_names=dm.target_cols,
            scaler_y=dm.scaler_y
        )

        # Évaluation et génération du rapport PDF
        evaluator = Evaluator(model, dm, config)
        evaluator.evaluate()

    except Exception as e:
        print("Erreur critique dans le main:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
