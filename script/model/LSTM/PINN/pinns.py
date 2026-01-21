# pinns.py
"""
Moteur PINN plug-and-play.
Définis un interface simple: chaque PINN doit hériter de PINNBase et implémenter
compute_pinn_loss(batch_inputs, batch_preds, meta) -> torch.Tensor (scalable scalar).

- batch_inputs: numpy array ou torch tensor contenant features de l'entrée (batch, seq_len, n_features) ou (batch, n_features)
- batch_preds: torch tensor (batch, n_outputs) prédits par le modèle (déjà sur device)
- meta: dictionnaire fourni par le pipeline, contient :
    - 'feature_cols': liste de noms de features (dans l'ordre utilisé par le pipeline)
    - 'target_names': liste des noms cibles (ordre)
    - 'scaler_y' (optionnel) : StandardScaler utilisé pour y (si le PINN veut inverser le scaling)
    - 'device' : torch.device

Exemple fourni: CosSinPINN -> pénalise (v1**2 + v2**2 - 1)**2 pour une paire v1/v2.
Peut opérer soit sur les targets (in_targets=True) soit sur les features (in_targets=False).
"""

import torch
import numpy as np
import torch.nn as nn

class PINNBase:
    def __init__(self, weight=1.0):
        self.weight = float(weight)
    def compute_pinn_loss(self, batch_inputs, batch_preds, meta):
        """
        Retourne un scalaire torch (loss) ou None si aucun terme PINN.
        """
        raise NotImplementedError("Implement in subclass")

class CosSinPINN(PINNBase):
    """
    Enforce v1^2 + v2^2 ~= 1 for a specified pair of variable names.
    Params:
        var1_name, var2_name: names of variables (strings)
        in_targets: if True, the variables are expected in the predicted outputs (batch_preds)
                    if False, the variables are expected in the inputs features (batch_inputs)
        weight: multiplicative weight for the PINN loss
    """
    def __init__(self, var1_name, var2_name, in_targets=False, weight=1.0):
        super().__init__(weight=weight)
        self.var1 = var1_name
        self.var2 = var2_name
        self.in_targets = bool(in_targets)

    def _get_indices(self, names_list):
        # return indices of var1 and var2 in names_list (or (-1,-1) if missing)
        try:
            i1 = names_list.index(self.var1)
            i2 = names_list.index(self.var2)
            return i1, i2
        except ValueError:
            return None, None

    def compute_pinn_loss(self, batch_inputs, batch_preds, meta):
        device = meta.get('device', torch.device('cpu'))
        feature_cols = meta.get('feature_cols', [])
        target_names = meta.get('target_names', [])

        if self.in_targets:
            # operate on batch_preds (torch tensor shape (batch, n_outputs))
            if batch_preds is None:
                return None
            i1, i2 = self._get_indices(target_names)
            if i1 is None or i2 is None:
                # missing variables in targets
                return None
            # batch_preds is torch tensor
            v1 = batch_preds[:, i1]
            v2 = batch_preds[:, i2]
            # enforce v1**2 + v2**2 ~ 1
            res = (v1**2 + v2**2 - 1.0)**2
            loss = torch.mean(res) * self.weight
            return loss
        else:
            # operate on features in batch_inputs (numpy array or torch tensor)
            if batch_inputs is None:
                return None
            i1, i2 = self._get_indices(feature_cols)
            if i1 is None or i2 is None:
                return None
            # batch_inputs could be numpy (batch, seq_len, n_features) or (n_samples, n_features)
            # we compute over the last timestep if seq_len dimension exists
            if isinstance(batch_inputs, np.ndarray):
                arr = batch_inputs
                if arr.ndim == 3:
                    vals = arr[:, -1, :]  # last step features
                else:
                    vals = arr
                v1 = torch.tensor(vals[:, i1], dtype=torch.float32, device=device)
                v2 = torch.tensor(vals[:, i2], dtype=torch.float32, device=device)
            else:
                # assume torch tensor
                arr = batch_inputs
                if arr.dim() == 3:
                    vals = arr[:, -1, :]
                else:
                    vals = arr
                v1 = vals[:, i1].to(device)
                v2 = vals[:, i2].to(device)
            res = (v1**2 + v2**2 - 1.0)**2
            loss = torch.mean(res) * self.weight
            return loss

class DissolvedOxygenPINN(PINNBase, nn.Module):
    """
    PINN pour l'oxygène dissous
    """

    def __init__(
        self,
        do_name: str = "dissolved_oxygen (ml l-1)",
        temp_water_name: str = "temperature (°C)",  # eau
        temp_air_name: str = "Mean Temp (°C)",      # air
        chl_name: str = "chlorophyll (mg m-3)",
        wind_name: str = "Spd of Max Gust (km/h)",
        tide_name: str = "tide_range (m)",
        sal_name: str = "salinity (PSS-78)",
        weight: float = 1.0,
    ):
        PINNBase.__init__(self, weight=weight)
        nn.Module.__init__(self)

        self.do_name = do_name
        self.temp_water_name = temp_water_name
        self.temp_air_name = temp_air_name
        self.chl_name = chl_name
        self.wind_name = wind_name
        self.tide_name = tide_name
        self.sal_name = sal_name

        # -----------------------------
        # Paramètres physiques appris
        # -----------------------------
        self.alpha_P = nn.Parameter(torch.tensor(0.05))
        self.alpha_R = nn.Parameter(torch.tensor(0.03))
        self.beta_R  = nn.Parameter(torch.tensor(0.07))
        self.alpha_k = nn.Parameter(torch.tensor(0.1))
        self.gamma_air = nn.Parameter(torch.tensor(0.05))

    # ------------------------------------------------------------------
    # Weiss 1970 saturation DO
    # ------------------------------------------------------------------

    def do_saturation(self, T, S):
        """
        DO saturation in seawater (Weiss 1970)
        T : °C
        S : salinity (PSU)
        returns : ml/L
        """
        Tk = T + 273.15

        A1 = -173.4292
        A2 = 249.6339
        A3 = 143.3483
        A4 = -21.8492

        B1 = -0.033096
        B2 = 0.014259
        B3 = -0.0017000

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

        device = meta.get("device", torch.device("cpu"))
        feature_cols = meta.get("feature_cols", [])
        target_names = meta.get("target_names", [])

        # --------------------------------------------------
        # Activation du PINN uniquement si DO est une cible
        # --------------------------------------------------
        if self.do_name not in target_names:
            return None

        required_inputs = [
            self.do_name,
            self.temp_water_name,
            self.temp_air_name,
            self.chl_name,
            self.wind_name,
            self.tide_name,
            self.sal_name,
        ]

        if not all(name in feature_cols for name in required_inputs):
            print("Input missing for DissolvedOxygenPINN")
            return None

        # --------------------------------------------------
        # Indices dynamiques (plug-and-play)
        # --------------------------------------------------
        try:
            i_do = feature_cols.index(self.do_name)
            i_Tw = feature_cols.index(self.temp_water_name)
            i_Ta = feature_cols.index(self.temp_air_name)
            i_chl = feature_cols.index(self.chl_name)
            i_wind = feature_cols.index(self.wind_name)
            i_tide = feature_cols.index(self.tide_name)
            i_sal = feature_cols.index(self.sal_name)
        except ValueError:
            return None

        # --------------------------------------------------
        # Conversion batch_inputs → torch
        # --------------------------------------------------
        if not torch.is_tensor(batch_inputs):
            X = torch.tensor(batch_inputs, dtype=torch.float32, device=device)
        else:
            X = batch_inputs.to(device)

        # attendu : (batch, seq_len, n_features)
        if X.dim() != 3 or X.shape[1] < 2:
            return None

        # --------------------------------------------------
        # Extraction des variables
        # --------------------------------------------------
        DO = X[:, :, i_do]
        Tw = X[:, :, i_Tw]  # température de l'eau
        Ta = X[:, :, i_Ta]  # température de l'air
        Chl = torch.relu(X[:, :, i_chl])
        Wind = torch.relu(X[:, :, i_wind])
        Tide = torch.relu(X[:, :, i_tide])
        Sal = torch.relu(X[:, :, i_sal])

        # --------------------------------------------------
        # Dérivée temporelle (Δt = 1 jour)
        # --------------------------------------------------
        dDO_dt = DO[:, 1:] - DO[:, :-1]

        DO_m = DO[:, :-1]
        Tw_m = Tw[:, :-1]
        Ta_m = Ta[:, :-1]
        Chl_m = Chl[:, :-1]
        Sal_m = Sal[:, :-1]

        # --------------------------------------------------
        # Termes physiques
        # --------------------------------------------------

        # Photosynthèse (biologique, eau)
        P = self.alpha_P * Chl_m * torch.exp(0.07 * Tw_m)

        # Respiration (biologique, eau)
        R = self.alpha_R * torch.exp(self.beta_R * Tw_m)

        # Réaération air–eau
        delta_T = torch.abs(Ta_m - Tw_m)

        krea = self.alpha_k * (
                Wind[:, :-1] + 0.5 * Tide[:, :-1]
        ) * (1.0 + self.gamma_air * delta_T)

        # Saturation O₂ (eau salée)
        DO_sat = self.do_saturation(Tw_m, Sal_m)

        # --------------------------------------------------
        # Résidu PINN
        # --------------------------------------------------
        rhs = P - R + krea * (DO_sat - DO_m)
        residual = dDO_dt - rhs

        return torch.mean(residual ** 2) * self.weight

class pHPINN(PINNBase, nn.Module):
    """
    PINN biogéochimique pour le pH
    Basé sur dynamique du carbone inorganique dissous (DIC)
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

        self.ph_name = ph_name
        self.temp_water_name = temp_water_name
        self.sal_name = sal_name
        self.chl_name = chl_name
        self.do_name = do_name
        self.wind_name = wind_name
        self.tide_name = tide_name

        # -----------------------------
        # Paramètres biogéochimiques appris
        # -----------------------------
        self.alpha_P = nn.Parameter(torch.tensor(0.03))   # photosynthèse
        self.alpha_R = nn.Parameter(torch.tensor(0.02))   # respiration (× f(T) × Chl)
        self.beta_R = nn.Parameter(torch.tensor(0.05))    # sensibilité T de la respiration
        self.alpha_k = nn.Parameter(torch.tensor(0.05))   # échange air–mer

        # pCO2 atmosphérique (µatm)
        self.pCO2_air = nn.Parameter(torch.tensor(420.0))

    # ------------------------------------------------------------------
    # Alcalinité totale (µmol/kg)
    # /!\ : techniquement pour une valeur de S comprise entre 31 et 37 
    #       et une valeur de T comprise entre 0 et 20 (partiellement respecté)
    # ------------------------------------------------------------------
    def total_alkalinity(self, T, S):
        return (
            2305
            + 53.97 * (S - 35.0)
            + 2.74 * (S - 35.0) ** 2
            - 1.16 * (T - 20.0)
            - 0.040 * (T - 20.0) ** 2
        )

    # ------------------------------------------------------------------
    # CO₂ dissous — Weiss 1974
    # ------------------------------------------------------------------
    def co2_solubility(self, T, S):
        Tk = T + 273.15

        A1 = -58.0931
        A2 = 90.5069
        A3 = 22.2940
        B1 = 0.027766
        B2 = -0.025888
        B3 = 0.0050578

        lnK0 = (
            A1
            + A2 * (100.0 / Tk)
            + A3 * torch.log(Tk / 100.0)
            + S * (B1 + B2 * (Tk / 100.0) + B3 * (Tk / 100.0) ** 2)
        )

        return torch.exp(lnK0)  # mol / L / atm

    # ------------------------------------------------------------------
    # K1, K2 dépendants de T et S (système carbonate, échelle mol/L)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # K1, K2 Lueker et al. (2000)
    # T en Celsius, S en PSU
    # Retourne K1, K2 sur l'échelle totale (mol/kg)
    # ------------------------------------------------------------------
    def k1_k2(self, T, S):
        Tk = T + 273.15
        lnT = torch.log(Tk)

        # pK1 (Lueker 2000)
        pK1 = (
            3633.86 / Tk
            - 61.2172
            + 9.67770 * lnT
            - 0.011555 * S
            + 0.0001152 * (S ** 2)
        )
        # pK2 (Lueker 2000)
        pK2 = (
            471.78 / Tk
            + 25.9290
            - 3.16967 * lnT
            - 0.01781 * S
            + 0.0001122 * (S ** 2)
        )
        
        K1 = torch.pow(10.0, -pK1)
        K2 = torch.pow(10.0, -pK2)
        return K1, K2

    # ------------------------------------------------------------------
    # Alcalinité des Borates (Dickson 1990 pour KB)
    # TB basé sur ratio 0.1324 mg/kg/PSU
    # ------------------------------------------------------------------
    def borate_alkalinity(self, T, S, H):
        Tk = T + 273.15
        lnT = torch.log(Tk)
        sqrtS = torch.sqrt(S)

        # KB - Dickson (1990)
        # lnKB
        lnKB = (
            (-8966.90 - 2890.53 * sqrtS - 77.942 * S + 1.728 * (S ** 1.5) - 0.0996 * (S ** 2)) / Tk
            + (148.0248 + 137.1942 * sqrtS + 1.62142 * S)
            + (-24.4344 - 25.085 * sqrtS - 0.2474 * S) * lnT
            + 0.053105 * sqrtS * Tk
        )
        KB = torch.exp(lnKB)

        # Total Boron (TB) en mol/kg
        # Ratio : 0.1324 mg/kg per salinity unit
        # M_Boron = 10.811 g/mol = 10811 mg/mol
        # TB = (0.1324 * S) / 10811
        TB = (0.1324 * S) / 10811.0
        
        # Borate Alkalinity: BA = TB * KB / (KB + H)
        BA = TB * KB / (KB + H)
        return BA

    # ------------------------------------------------------------------
    # DIC depuis TA et pH ( [H+] )
    # Prend en compte les borates : TA_C = TA - BA
    # DIC = TA_C * ( [H+]/K1 + 1 + K2/[H+] ) / (1 + 2*K2/[H+] )
    # TA en µmol/kg -> converti en mol/kg pour le calcul interne
    # ------------------------------------------------------------------
    def dic_from_ph_ta(self, TA, H, K1, K2, T=None, S=None):
        # TA est en µmol (~2000-2500), on le passe en mol pour la chimie
        TA_mol = TA * 1e-6
        H = torch.clamp(H, min=1e-12, max=1.0)
        
        # Correction Borates si T et S fournis (ce qui devrait être le cas)
        if T is not None and S is not None:
            BA = self.borate_alkalinity(T, S, H)
            # Alcalinité Carbonate
            CA_mol = TA_mol - BA
        else:
            # Fallback sans borates (pas recommandé avec la nouvelle logique)
            CA_mol = TA_mol

        denom = torch.clamp(1.0 + 2.0 * K2 / H, min=1e-6)
        DIC_mol = CA_mol * (H / K1 + 1.0 + K2 / H) / denom
        
        # Retour en µmol/kg pour cohérence avec le reste du code ?
        # Dans le code original, dic_from_ph_ta retournait DIC (même unité que TA).
        # On retourne en µmol
        return DIC_mol * 1e6

    # ------------------------------------------------------------------
    # [CO2] depuis DIC et [H+] — [CO2] = DIC_mol / (1 + K1/[H+] + K1*K2/[H+]^2)
    # DIC_umol en µmol/L ; retour en mol/L
    # ------------------------------------------------------------------
    def co2_from_dic_h(self, DIC_umol, H, K1, K2):
        H = torch.clamp(H, min=1e-12, max=1.0)
        DIC_mol = DIC_umol * 1e-6
        denom = torch.clamp(1.0 + K1 / H + K1 * K2 / (H ** 2), min=1e-10)
        return DIC_mol / denom

    # ------------------------------------------------------------------
    # pH depuis DIC et TA (forward, différentiable) — approximation
    # L’inverse cohérent est dic_from_ph_ta + système carbonate.
    # ------------------------------------------------------------------
    def ph_from_dic(self, DIC, TA, T, S):
        K1, K2 = self.k1_k2(T, S)
        # [CO2] ≈ DIC / (1 + K1/H + K1*K2/H^2) ; TA ≈ [HCO3-] + 2[CO3--] => H ~ K1*DIC/TA (approx)
        H = torch.clamp(K1 * DIC / (TA + 1e-6), min=1e-12, max=1.0)
        return -torch.log10(H)

    # ------------------------------------------------------------------
    # PINN loss
    # ------------------------------------------------------------------
    def compute_pinn_loss(self, batch_inputs, batch_preds, meta):

        if batch_inputs is None:
            return None

        device = meta.get("device", torch.device("cpu"))
        feature_cols = meta.get("feature_cols", [])
        target_names = meta.get("target_names", [])

        # activation uniquement si pH est prédit
        if self.ph_name not in target_names:
            return None

        required = [
            self.ph_name,
            self.temp_water_name,
            self.sal_name,
            self.chl_name,
            self.wind_name,
            self.tide_name,
        ]

        if not all(v in feature_cols for v in required):
            return None

        try:
            i_ph_feat = feature_cols.index(self.ph_name)
            i_T = feature_cols.index(self.temp_water_name)
            i_S = feature_cols.index(self.sal_name)
            i_chl = feature_cols.index(self.chl_name)
            i_wind = feature_cols.index(self.wind_name)
            i_tide = feature_cols.index(self.tide_name)
            i_ph = target_names.index(self.ph_name)
        except ValueError:
            return None

        # tensor
        if not torch.is_tensor(batch_inputs):
            X = torch.tensor(batch_inputs, dtype=torch.float32, device=device)
        else:
            X = batch_inputs.to(device)

        if X.dim() != 3 or X.shape[1] < 1:
            return None

        # ------------------------------------------------------------------
        # Dernier pas de la séquence uniquement (transition → pas prédit)
        # ------------------------------------------------------------------
        T_last = X[:, -1, i_T]
        S_last = torch.relu(X[:, -1, i_S])
        Chl_last = torch.relu(X[:, -1, i_chl])
        Wind_last = torch.relu(X[:, -1, i_wind])
        Tide_last = torch.relu(X[:, -1, i_tide])
        pH_prev = X[:, -1, i_ph_feat]
        ph_pred = batch_preds[:, i_ph]

        H_prev = torch.clamp(torch.pow(10.0, -pH_prev), min=1e-12, max=1.0)
        H_pred = torch.clamp(torch.pow(10.0, -ph_pred), min=1e-12, max=1.0)

        TA = self.total_alkalinity(T_last, S_last)
        K1, K2 = self.k1_k2(T_last, S_last)

        # DIC à t (input) et à t+1 (prédit) — formules carbonate cohérentes
        DIC_prev = self.dic_from_ph_ta(TA, H_prev, K1, K2, T=T_last, S=S_last)
        DIC_pred = self.dic_from_ph_ta(TA, H_pred, K1, K2, T=T_last, S=S_last)

        # dDIC/dt entre dernier pas d’entrée et pas prédit (Δt = 1 jour)
        dDIC_dt = DIC_pred - DIC_prev

        # -----------------------------
        # Termes biogéochimiques (µmol/L/jour)
        # -----------------------------
        P = self.alpha_P * Chl_last
        R = self.alpha_R * torch.exp(self.beta_R * T_last) * Chl_last

        # Flux air–mer : F = 1e6 * k_gas * ([CO2]_eq - [CO2]_sea) en µmol/L/jour
        K0 = self.co2_solubility(T_last, S_last)
        pCO2_atm = torch.clamp(self.pCO2_air, min=200.0, max=600.0) * 1e-6  # µatm → atm
        CO2_eq = K0 * pCO2_atm
        CO2_sea = self.co2_from_dic_h(DIC_prev, H_prev, K1, K2)

        k_gas = self.alpha_k * (Wind_last + 0.5 * Tide_last + 1e-6)
        F = 1e6 * k_gas * (CO2_eq - CO2_sea)

        rhs = -P + R + F
        residual = dDIC_dt - rhs

        return torch.mean(residual ** 2) * self.weight
