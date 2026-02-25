# pinns.py
"""
Moteur PINN modulaire (plug-and-play).
Chaque contrainte physique hérite de PINNBase et implémente
compute_pinn_loss(batch_inputs, batch_preds, meta) -> torch.Tensor.

Arguments reçus par compute_pinn_loss :
  - batch_inputs : tableau numpy ou tenseur torch (batch, seq_len, n_features)
  - batch_preds  : tenseur torch (batch, n_outputs) — prédictions du modèle
  - meta         : dictionnaire fourni par le Trainer, contenant :
      'feature_cols' : liste ordonnée des noms de features
      'target_names' : liste ordonnée des noms de cibles
      'scaler_y'     : StandardScaler utilisé sur y (optionnel)
      'device'       : torch.device

Contraintes disponibles :
  - CosSinPINN         : pénalise (v1² + v2² - 1)² pour une paire (sin, cos)
  - DissolvedOxygenPINN: bilan O₂ dissous (photosynthèse, respiration, réaération)
  - pHPINN             : bilan DIC / pH (carbonate, borates, flux air-mer)
"""

import torch
import numpy as np
import torch.nn as nn


class PINNBase:
    def __init__(self, weight=1.0):
        self.weight = float(weight)

    def compute_pinn_loss(self, batch_inputs, batch_preds, meta):
        """Retourne un scalaire torch (loss) ou None si la contrainte ne s'applique pas."""
        raise NotImplementedError("À implémenter dans la sous-classe")


class CosSinPINN(PINNBase):
    """
    Contraint v1² + v2² ≈ 1 pour une paire de variables (ex. doy_sin / doy_cos).
    Peut opérer sur les features d'entrée (in_targets=False) ou sur les prédictions (in_targets=True).
    """
    def __init__(self, var1_name, var2_name, in_targets=False, weight=1.0):
        super().__init__(weight=weight)
        self.var1 = var1_name
        self.var2 = var2_name
        self.in_targets = bool(in_targets)

    def _get_indices(self, names_list):
        """Retourne les indices de var1 et var2 dans names_list, ou (None, None) si absents."""
        try:
            return names_list.index(self.var1), names_list.index(self.var2)
        except ValueError:
            return None, None

    def compute_pinn_loss(self, batch_inputs, batch_preds, meta):
        device = meta.get('device', torch.device('cpu'))
        feature_cols = meta.get('feature_cols', [])
        target_names = meta.get('target_names', [])

        if self.in_targets:
            if batch_preds is None:
                return None
            i1, i2 = self._get_indices(target_names)
            if i1 is None or i2 is None:
                return None
            v1 = batch_preds[:, i1]
            v2 = batch_preds[:, i2]
        else:
            if batch_inputs is None:
                return None
            i1, i2 = self._get_indices(feature_cols)
            if i1 is None or i2 is None:
                return None
            # Extraction du dernier pas temporel si batch_inputs est 3D
            if isinstance(batch_inputs, np.ndarray):
                arr = batch_inputs
                vals = arr[:, -1, :] if arr.ndim == 3 else arr
                v1 = torch.tensor(vals[:, i1], dtype=torch.float32, device=device)
                v2 = torch.tensor(vals[:, i2], dtype=torch.float32, device=device)
            else:
                arr = batch_inputs
                vals = arr[:, -1, :] if arr.dim() == 3 else arr
                v1 = vals[:, i1].to(device)
                v2 = vals[:, i2].to(device)

        return torch.mean((v1**2 + v2**2 - 1.0)**2) * self.weight


class DissolvedOxygenPINN(PINNBase, nn.Module):
    """
    Contrainte physique sur l'oxygène dissous (O₂).
    Modélise le bilan journalier : dDO/dt ≈ Photosynthèse - Respiration + Réaération.
    Les coefficients biogéochimiques (alpha_P, alpha_R, beta_R, alpha_k, gamma_air)
    sont des paramètres appris par rétropropagation.
    """

    def __init__(
        self,
        do_name: str = "dissolved_oxygen (ml l-1)",
        temp_water_name: str = "temperature (°C)",
        temp_air_name: str = "Mean Temp (°C)",
        chl_name: str = "chlorophyll (mg m-3)",
        wind_name: str = "Spd of Max Gust (km/h)",
        tide_name: str = "tide_range (m)",
        sal_name: str = "salinity (PSS-78)",
        weight: float = 1.0,
    ):
        PINNBase.__init__(self, weight=weight)
        nn.Module.__init__(self)

        self.do_name         = do_name
        self.temp_water_name = temp_water_name
        self.temp_air_name   = temp_air_name
        self.chl_name        = chl_name
        self.wind_name       = wind_name
        self.tide_name       = tide_name
        self.sal_name        = sal_name

        # Paramètres biogéochimiques appris
        self.alpha_P   = nn.Parameter(torch.tensor(0.05))   # taux de photosynthèse
        self.alpha_R   = nn.Parameter(torch.tensor(0.03))   # taux de respiration
        self.beta_R    = nn.Parameter(torch.tensor(0.07))   # sensibilité T de la respiration
        self.alpha_k   = nn.Parameter(torch.tensor(0.1))    # coefficient d'échange air-eau
        self.gamma_air = nn.Parameter(torch.tensor(0.05))   # sensibilité à l'écart T air-eau

    def do_saturation(self, T, S):
        """
        Saturation en O₂ dissous en eau de mer (Weiss 1970).
        T en °C, S en PSU. Retourne DO_sat en ml/L.
        """
        Tk = T + 273.15
        A1, A2, A3, A4 = -173.4292, 249.6339, 143.3483, -21.8492
        B1, B2, B3     = -0.033096, 0.014259, -0.0017000

        lnC = (
            A1
            + A2 * (100.0 / Tk)
            + A3 * torch.log(Tk / 100.0)
            + A4 * (Tk / 100.0)
            + S * (B1 + B2 * (Tk / 100.0) + B3 * (Tk / 100.0) ** 2)
        )
        return torch.exp(lnC)

    def compute_pinn_loss(self, batch_inputs, batch_preds, meta):
        if batch_inputs is None:
            return None

        device       = meta.get("device", torch.device("cpu"))
        feature_cols = meta.get("feature_cols", [])
        target_names = meta.get("target_names", [])

        if self.do_name not in target_names:
            return None

        required = [self.do_name, self.temp_water_name, self.temp_air_name,
                    self.chl_name, self.wind_name, self.tide_name, self.sal_name]
        if not all(name in feature_cols for name in required):
            print("DissolvedOxygenPINN : feature manquante. Skip.")
            return None

        try:
            i_do   = feature_cols.index(self.do_name)
            i_Tw   = feature_cols.index(self.temp_water_name)
            i_Ta   = feature_cols.index(self.temp_air_name)
            i_chl  = feature_cols.index(self.chl_name)
            i_wind = feature_cols.index(self.wind_name)
            i_tide = feature_cols.index(self.tide_name)
            i_sal  = feature_cols.index(self.sal_name)
        except ValueError:
            return None

        X = torch.tensor(batch_inputs, dtype=torch.float32, device=device) \
            if not torch.is_tensor(batch_inputs) else batch_inputs.to(device)

        # Nécessite au moins 2 pas de temps pour calculer dDO/dt
        if X.dim() != 3 or X.shape[1] < 2:
            return None

        DO   = X[:, :, i_do]
        Tw   = X[:, :, i_Tw]
        Ta   = X[:, :, i_Ta]
        Chl  = torch.relu(X[:, :, i_chl])
        Wind = torch.relu(X[:, :, i_wind])
        Tide = torch.relu(X[:, :, i_tide])
        Sal  = torch.relu(X[:, :, i_sal])

        # Dérivée temporelle discrète (Δt = 1 jour)
        dDO_dt = DO[:, 1:] - DO[:, :-1]

        DO_m, Tw_m, Ta_m = DO[:, :-1], Tw[:, :-1], Ta[:, :-1]
        Chl_m, Sal_m     = Chl[:, :-1], Sal[:, :-1]

        # Termes du bilan O₂
        P     = self.alpha_P * Chl_m * torch.exp(0.07 * Tw_m)          # photosynthèse
        R     = self.alpha_R * torch.exp(self.beta_R * Tw_m)            # respiration
        delta_T = torch.abs(Ta_m - Tw_m)
        krea  = self.alpha_k * (Wind[:, :-1] + 0.5 * Tide[:, :-1]) * (1.0 + self.gamma_air * delta_T)
        DO_sat = self.do_saturation(Tw_m, Sal_m)

        rhs      = P - R + krea * (DO_sat - DO_m)
        residual = dDO_dt - rhs
        return torch.mean(residual ** 2) * self.weight


class pHPINN(PINNBase, nn.Module):
    """
    Contrainte biogéochimique sur le pH basée sur la dynamique du carbone inorganique dissous (DIC).
    Inclut : alcalinité (Lee et al.), solubilité CO₂ (Weiss 1974), K1/K2 (Lueker 2000),
    alcalinité des borates (Dickson 1990) et flux air-mer.
    """

    def __init__(
        self,
        ph_name: str = "pH",
        temp_water_name: str = "temperature (°C)",
        sal_name: str = "salinity (PSS-78)",
        chl_name: str = "chlorophyll (mg m-3)",
        do_name: str = "dissolved_oxygen (ml l-1)",
        wind_name: str = "Spd of Max Gust (km/h)",
        tide_name: str = "tide_range (m)",
        weight: float = 1.0,
    ):
        PINNBase.__init__(self, weight=weight)
        nn.Module.__init__(self)

        self.ph_name         = ph_name
        self.temp_water_name = temp_water_name
        self.sal_name        = sal_name
        self.chl_name        = chl_name
        self.do_name         = do_name
        self.wind_name       = wind_name
        self.tide_name       = tide_name

        # Paramètres biogéochimiques appris
        self.alpha_P  = nn.Parameter(torch.tensor(0.03))    # taux de photosynthèse
        self.alpha_R  = nn.Parameter(torch.tensor(0.02))    # taux de respiration
        self.beta_R   = nn.Parameter(torch.tensor(0.05))    # sensibilité T de la respiration
        self.alpha_k  = nn.Parameter(torch.tensor(0.05))    # coefficient d'échange air-mer
        self.pCO2_air = nn.Parameter(torch.tensor(420.0))   # pCO2 atmosphérique (µatm)

    def total_alkalinity(self, T, S):
        """
        Alcalinité totale (µmol/kg) — Lee et al.
        Valide pour S ∈ [31, 37] et T ∈ [0, 20] °C (partiellement respecté ici).
        """
        return (
            2305
            + 53.97 * (S - 35.0)
            + 2.74  * (S - 35.0) ** 2
            - 1.16  * (T - 20.0)
            - 0.040 * (T - 20.0) ** 2
        )

    def co2_solubility(self, T, S):
        """Solubilité du CO₂ en eau de mer — Weiss (1974). Retourne K0 en mol/L/atm."""
        Tk = T + 273.15
        A1, A2, A3 = -58.0931, 90.5069, 22.2940
        B1, B2, B3 =  0.027766, -0.025888, 0.0050578

        lnK0 = (
            A1
            + A2 * (100.0 / Tk)
            + A3 * torch.log(Tk / 100.0)
            + S * (B1 + B2 * (Tk / 100.0) + B3 * (Tk / 100.0) ** 2)
        )
        return torch.exp(lnK0)

    def k1_k2(self, T, S):
        """
        Constantes de dissociation du système carbonate — Lueker et al. (2000).
        T en °C, S en PSU. Retourne K1 et K2 sur l'échelle totale (mol/kg).
        """
        Tk  = T + 273.15
        lnT = torch.log(Tk)

        pK1 = (3633.86 / Tk - 61.2172 + 9.67770 * lnT - 0.011555 * S + 0.0001152 * (S ** 2))
        pK2 = (471.78  / Tk + 25.9290 - 3.16967 * lnT - 0.01781  * S + 0.0001122 * (S ** 2))

        return torch.pow(10.0, -pK1), torch.pow(10.0, -pK2)

    def borate_alkalinity(self, T, S, H):
        """
        Alcalinité des borates — Dickson (1990) pour KB, ratio TB = 0.1324 mg/kg/PSU.
        Retourne BA en mol/kg.
        """
        Tk    = T + 273.15
        lnT   = torch.log(Tk)
        sqrtS = torch.sqrt(S)

        lnKB = (
            (-8966.90 - 2890.53 * sqrtS - 77.942 * S + 1.728 * (S ** 1.5) - 0.0996 * (S ** 2)) / Tk
            + (148.0248 + 137.1942 * sqrtS + 1.62142 * S)
            + (-24.4344 - 25.085 * sqrtS - 0.2474 * S) * lnT
            + 0.053105 * sqrtS * Tk
        )
        KB = torch.exp(lnKB)
        TB = (0.1324 * S) / 10811.0   # bore total en mol/kg
        return TB * KB / (KB + H)

    def dic_from_ph_ta(self, TA, H, K1, K2, T=None, S=None):
        """
        DIC depuis l'alcalinité totale (µmol/kg) et [H+].
        Prend en compte l'alcalinité des borates si T et S sont fournis.
        Retourne DIC en µmol/kg.
        """
        TA_mol = TA * 1e-6
        H = torch.clamp(H, min=1e-12, max=1.0)

        if T is not None and S is not None:
            BA     = self.borate_alkalinity(T, S, H)
            CA_mol = TA_mol - BA
        else:
            CA_mol = TA_mol

        denom   = torch.clamp(1.0 + 2.0 * K2 / H, min=1e-6)
        DIC_mol = CA_mol * (H / K1 + 1.0 + K2 / H) / denom
        return DIC_mol * 1e6

    def co2_from_dic_h(self, DIC_umol, H, K1, K2):
        """[CO₂] dissous depuis DIC (µmol/L) et [H+]. Retourne en mol/L."""
        H       = torch.clamp(H, min=1e-12, max=1.0)
        DIC_mol = DIC_umol * 1e-6
        denom   = torch.clamp(1.0 + K1 / H + K1 * K2 / (H ** 2), min=1e-10)
        return DIC_mol / denom

    def ph_from_dic(self, DIC, TA, T, S):
        """
        Approximation du pH depuis DIC et TA.
        H ≈ K1 * DIC / TA (approximation carbonates uniquement).
        """
        K1, K2 = self.k1_k2(T, S)
        H = torch.clamp(K1 * DIC / (TA + 1e-6), min=1e-12, max=1.0)
        return -torch.log10(H)

    def compute_pinn_loss(self, batch_inputs, batch_preds, meta):
        if batch_inputs is None:
            return None

        device       = meta.get("device", torch.device("cpu"))
        feature_cols = meta.get("feature_cols", [])
        target_names = meta.get("target_names", [])

        if self.ph_name not in target_names:
            return None

        required = [self.ph_name, self.temp_water_name, self.sal_name,
                    self.chl_name, self.wind_name, self.tide_name]
        if not all(v in feature_cols for v in required):
            return None

        try:
            i_ph_feat = feature_cols.index(self.ph_name)
            i_T       = feature_cols.index(self.temp_water_name)
            i_S       = feature_cols.index(self.sal_name)
            i_chl     = feature_cols.index(self.chl_name)
            i_wind    = feature_cols.index(self.wind_name)
            i_tide    = feature_cols.index(self.tide_name)
            i_ph      = target_names.index(self.ph_name)
        except ValueError:
            return None

        X = torch.tensor(batch_inputs, dtype=torch.float32, device=device) \
            if not torch.is_tensor(batch_inputs) else batch_inputs.to(device)

        if X.dim() != 3 or X.shape[1] < 1:
            return None

        # Extraction au dernier pas de la séquence (transition vers le pas prédit)
        T_last    = X[:, -1, i_T]
        S_last    = torch.relu(X[:, -1, i_S])
        Chl_last  = torch.relu(X[:, -1, i_chl])
        Wind_last = torch.relu(X[:, -1, i_wind])
        Tide_last = torch.relu(X[:, -1, i_tide])
        pH_prev   = X[:, -1, i_ph_feat]
        ph_pred   = batch_preds[:, i_ph]

        H_prev = torch.clamp(torch.pow(10.0, -pH_prev), min=1e-12, max=1.0)
        H_pred = torch.clamp(torch.pow(10.0, -ph_pred), min=1e-12, max=1.0)

        TA      = self.total_alkalinity(T_last, S_last)
        K1, K2  = self.k1_k2(T_last, S_last)

        # DIC à t (entrée) et t+1 (prédit) via les équations du carbonate
        DIC_prev = self.dic_from_ph_ta(TA, H_prev, K1, K2, T=T_last, S=S_last)
        DIC_pred = self.dic_from_ph_ta(TA, H_pred, K1, K2, T=T_last, S=S_last)

        dDIC_dt = DIC_pred - DIC_prev   # Δt = 1 jour

        # Termes biogéochimiques (µmol/L/jour)
        P = self.alpha_P * Chl_last
        R = self.alpha_R * torch.exp(self.beta_R * T_last) * Chl_last

        # Flux air-mer de CO₂
        K0       = self.co2_solubility(T_last, S_last)
        pCO2_atm = torch.clamp(self.pCO2_air, min=200.0, max=600.0) * 1e-6  # µatm → atm
        CO2_eq   = K0 * pCO2_atm
        CO2_sea  = self.co2_from_dic_h(DIC_prev, H_prev, K1, K2)
        k_gas    = self.alpha_k * (Wind_last + 0.5 * Tide_last + 1e-6)
        F        = 1e6 * k_gas * (CO2_eq - CO2_sea)

        rhs      = -P + R + F
        residual = dDIC_dt - rhs
        return torch.mean(residual ** 2) * self.weight
