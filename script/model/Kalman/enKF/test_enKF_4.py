# fichier: enkf_timeseries_fixed_with_pdf_depth_filtered_daily_agg.py
"""
Version modifiée: poids par composante pour l'estimation de A.
- Train rééchantillonné journalier (moyenne ou médiane par jour au choix)
- Test: observations laissées inchangées (sparse) mais on calcule aussi l'agrégation journalière pour l'affichage/metrics
- FEATURE_WEIGHTS / TARGET_WEIGHTS permettent d'accentuer/ignorer des composantes
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

# sélection autour d'une profondeur cible ---
DEPTH_COL = "depth"
DEPTH_TARGET = 1
DEPTH_TOL = 0.1

# NEW TUNABLE FLAGS
USE_SEASONAL = True
N_HARMONICS = 3
USE_INTERCEPT = True
ASSIMILATE = False
Q_scale = 0.0
R_scale = 1.0
RESAMPLE_DAILY = True

# Si True -> on utilisera la médiane pour l'entraînement et l'affichage/metrics,
# sinon la moyenne.
USE_DAILY_MEDIAN = False
mean_pdf = "Mean"
if USE_DAILY_MEDIAN:
    mean_pdf = "Median"

# Fréquence d'agrégation temporelle pour l'entraînement et l'affichage
# Choix possibles : 'D' (journalier), 'W' (hebdomadaire), 'M' (mensuel)
AGGREGATION_FREQ = 'D'


# ---------- CONFIG ----------
PARQUET_PATH = "../../dataset/OMDS-CTD datalight_with_pos.parquet"
TIME_COL = "time"
TRAIN_START = "2000-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2025-12-30"
ENS_SIZE = 100
Ridge_alpha = 1.0
SEED = 42
OUTPUT_PDF = "../results/prediction/enKF/v4/predictions_report_" + mean_pdf + "_" + AGGREGATION_FREQ + "_" + str(DEPTH_TARGET) + "m.pdf"

# ----------------------------
np.random.seed(SEED)
# ----------------------------

# -------------------------------------------------------------------------
# Weights config: ajuste ici l'importance de chaque composante pendant l'estimation
# Mettre 0 pour ignorer complètement une composante (elle sera retirée de l'état)
# Clés attendues : noms exacts dans var_cols_aug (p.ex. "TEMPS901", "TURBPR01", "sin_1", "days_since_start", ...)
FEATURE_WEIGHTS = {
    # Exemples: (ajuste selon tes besoins)
    "TEMPS901": 1.0,
    "CPHLPR01": 0.0,
    "TURBPR01": 0.0,
    "PHXXPR01": 0.0,
    "PSALST01": 0.0,
    "SIGTEQST": 0.0,
    "DOXYZZ01": 1.0,
    # saison et trend — tu peux augmenter pour forcer date forte
    "sin_1": 2.5, "cos_1": 2.5,
    "sin_2": 1.0, "cos_2": 1.0,
    "sin_3": 0.5, "cos_3": 0.5,
    "days_since_start": 2.0,
}
TARGET_WEIGHTS = {
    # Exemple: donner plus d'importance à la qualité de prédiction
    "TEMPS901": 1.0,
    "CPHLPR01": 0.0,
    "TURBPR01": 0.0,
    "PHXXPR01": 0.0,
    "PSALST01": 0.0,
    "SIGTEQST": 0.0,
    "DOXYZZ01": 1.0
}
# -------------------------------------------------------------------------

def read_and_clean(path):
    import time
    t0 = time.time()
    print("[INFO] Lecture du fichier parquet :", path)
    df = pd.read_parquet(path)

    print(f"[INFO] Lecture terminée en {time.time() - t0:.2f}s, {len(df):,} lignes, {len(df.columns)} colonnes")

    if "UTC" in df.columns and TIME_COL not in df.columns:
        df = df.rename(columns={"UTC": TIME_COL})

    if TIME_COL not in df.columns:
        raise KeyError(f"Colonne temporelle '{TIME_COL}' introuvable dans le fichier parquet.")

    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        print("[INFO] Conversion de la colonne time en datetime (format initial =", df[TIME_COL].dtype, ")")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
        n_bad = df[TIME_COL].isna().sum()
        if n_bad > 0:
            print(f"[INFO] Suppression de {n_bad} lignes avec time non parseable.")
            df = df.dropna(subset=[TIME_COL])

    df = df.set_index(TIME_COL).sort_index()
    print(f"[INFO] Nettoyage terminé en {time.time() - t0:.2f}s total")
    return df


def apply_qc_rule(df):
    print("[INFO] Application des règles QC")
    qc_cols = [c for c in df.columns if c.endswith("_QC")]
    var_cols = [c for c in df.columns if c not in qc_cols]
    print(f"[INFO] Colonnes QC détectées: {qc_cols}")
    for qc in qc_cols:
        var = qc[:-3]
        if var in df.columns:
            # accepte QC == 1 ou 5 comme valides
            mask_bad = df[qc].notna() & ((df[qc] != 1) & (df[qc] != 5))
            n_bad = mask_bad.sum()
            if n_bad > 0:
                print(f"[INFO] QC: {n_bad} valeurs marquées comme invalides pour '{var}' (colonne {qc}).")
                df.loc[mask_bad, var] = np.nan
    return df, var_cols


def resample_and_interpolate(df, agg_method='mean', freq=AGGREGATION_FREQ):
    """Resample à la fréquence donnée (journalière/hebdo/mensuelle) avec mean/median puis interpolation."""
    tmp = df.copy()
    try:
        tmp.index = tmp.index.tz_convert(None)
    except Exception:
        pass
    if agg_method == 'median':
        df_resampled = tmp.resample(freq).median()
    else:
        df_resampled = tmp.resample(freq).mean()
    df_resampled = df_resampled.interpolate(method="time", limit_direction="both")
    return df_resampled


def aggregate_obs(df, cols=None, agg_method='mean', freq=AGGREGATION_FREQ):
    """Agrège observations multiples selon la fréquence spécifiée ('D', 'W', 'M')."""
    if cols is None:
        cols = df.columns.tolist()
    tmp = df.copy()
    try:
        idx_naive = tmp.index.tz_convert(None)
    except Exception:
        idx_naive = tmp.index
    if len(tmp) == 0:
        return tmp.iloc[0:0]
    tmp = tmp.assign(__period=idx_naive.to_period(freq))
    if agg_method == 'median':
        agg = tmp.groupby('__period')[cols].median()
    else:
        agg = tmp.groupby('__period')[cols].mean()
    agg.index = agg.index.to_timestamp()
    return agg



def add_seasonal_harmonics(df, n_harmonics=3):
    tmp = df.copy()
    try:
        idx_naive = tmp.index.tz_convert(None)
    except Exception:
        idx_naive = tmp.index
    doy = idx_naive.dayofyear.values.astype(float)
    for k in range(1, n_harmonics+1):
        ang = 2.0 * np.pi * k * (doy / 365.25)
        tmp[f"sin_{k}"] = np.sin(ang)
        tmp[f"cos_{k}"] = np.cos(ang)
    return tmp


def add_days_since_start(df):
    tmp = df.copy()
    try:
        idx_naive = tmp.index.tz_convert(None)
    except Exception:
        idx_naive = tmp.index
    if len(idx_naive) == 0:
        tmp["days_since_start"] = np.nan
        return tmp
    days = (idx_naive - idx_naive[0]).total_seconds() / (24*3600.0)
    tmp["days_since_start"] = days
    return tmp


def build_training_pairs(X):
    print("[INFO] Construction des paires d'entraînement (x_t, x_{t+1})")
    Xs = []
    Xn = []
    for i in range(len(X)-1):
        xt = X.iloc[i].values
        xt1 = X.iloc[i+1].values
        if np.any(np.isnan(xt)) or np.any(np.isnan(xt1)):
            continue
        Xs.append(xt)
        Xn.append(xt1)
    if len(Xs) == 0:
        raise ValueError("Pas de paires (x_t,x_{t+1}) valides pour l'entraînement.")
    Xs = np.vstack(Xs)
    Xn = np.vstack(Xn)
    print(f"[INFO] {Xs.shape[0]} paires construites, dimension d'état = {Xs.shape[1]}")
    return Xs, Xn

# estimation pondérée de A
def estimate_A_weighted(Xs, Xn, var_cols, feature_weights=None, target_weights=None, alpha=1.0, fit_intercept=True, eps_small=1e-8):
    """
    Version multioutput + suppression de variables si s==0.
    Même API / mêmes sorties que la fonction d'origine :
      inputs:
        Xs: array (N, p) - features at step s
        Xn: array (N, p) - targets at step n
        var_cols: list of length p, names aligned with columns
        feature_weights: dict name->float (appliqué aux colonnes de Xs)
        target_weights: dict name->float (appliqué aux colonnes de Xn)
        alpha, fit_intercept: passed to sklearn.linear_model.Ridge
        eps_small: valeur minimale pour t (éviter division par zéro)
      returns:
        A: (p, p) matrix
        b: (p,) vector
        Q: (p, p) covariance matrix of residuals
        resid: (N, p) residuals in original scale (Xn - Xs @ A.T - b)
    Comportement important:
      - si feature_weights[name] == 0 -> la colonne est SUPPRIMEE de Xs au fit;
        les coefficients correspondants dans A seront mis à zéro.
      - target_weights petits/égaux à 0 sont remplacés par eps_small (évite divisions)
    """
    print("[INFO] Estimation de A pondéré (multioutput) fit_intercept=%s" % fit_intercept)

    # vérifs basiques
    Xs = np.asarray(Xs)
    Xn = np.asarray(Xn)
    n_vars = Xs.shape[1]
    if len(var_cols) != n_vars:
        raise ValueError("var_cols length inconsistent with Xs shape.")
    if Xn.shape[1] != n_vars:
        raise ValueError("Xn must have same number of columns as Xs.")

    feature_weights = feature_weights or {}
    target_weights = target_weights or {}

    # construire vecteurs s et t alignés sur var_cols
    s = np.ones(n_vars, dtype=float)
    t = np.ones(n_vars, dtype=float)
    for i, name in enumerate(var_cols):
        if name in feature_weights:
            try:
                s[i] = float(feature_weights[name])
            except Exception:
                s[i] = 1.0
        if name in target_weights:
            try:
                t[i] = float(target_weights[name])
            except Exception:
                t[i] = 1.0

    # trouver les colonnes à garder dans Xs (s != 0)
    keep_mask = (s != 0.0)
    removed_cols = np.where(~keep_mask)[0]
    if removed_cols.size > 0:
        print(f"[INFO] Removing {removed_cols.size} feature column(s) because s==0: indices {removed_cols.tolist()}")

    # protéger t contre 0 (on remplace par eps_small pour garder les opérations stables)
    zero_t = (t == 0.0)
    if np.any(zero_t):
        print(f"[WARN] Some target weights are zero; replacing by eps_small={eps_small} to avoid division by zero.")
        t[zero_t] = eps_small

    # mise à l'échelle : pour les features gardées on multiplie, pour celes enlevées on ne les inclut pas
    if keep_mask.sum() > 0:
        Xs_scaled = Xs[:, keep_mask] * s[keep_mask].reshape(1, -1)
    else:
        Xs_scaled = np.zeros((Xs.shape[0], 0))  # pas de features après suppression

    Xn_scaled = Xn * t.reshape(1, -1)

    N = Xs.shape[0]

    # cas particulier : si aucune feature gardée -> on ne peut pas fit de A, on calcule seulement b (intercept)
    n_kept = Xs_scaled.shape[1]
    A_scaled_full = np.zeros((n_vars, n_kept))  # stocke les coefs sur features gardées (rows: target j)
    b_scaled = np.zeros(n_vars)

    if n_kept == 0:
        # pas de features, on ne peut qu'estimer l'intercept (moyenne des targets si fit_intercept)
        if fit_intercept:
            # L'estimateur d'intercept en Ridge multioutput sans features = moyenne des y (comme sklearn le ferait)
            b_scaled = np.mean(Xn_scaled, axis=0)
        else:
            b_scaled = np.zeros(n_vars)
        # A_scaled_full reste à zéro
    else:
        # Fit multioutput Ridge en une passe: Xs_scaled (N, n_kept) -> Xn_scaled (N, n_vars)
        # sklearn attend Y shape (N, n_targets)
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        # fit Y = Xs_scaled @ W.T + intercept
        model.fit(Xs_scaled, Xn_scaled)  # sklearn fit_multioutput: coef_ shape (n_targets, n_features)
        # récupérer coef & intercept
        # model.coef_ a shape (n_targets, n_features) i.e. (n_vars, n_kept)
        A_scaled_full = np.asarray(model.coef_)  # (n_vars, n_kept)
        if fit_intercept:
            b_scaled = np.asarray(model.intercept_)
        else:
            b_scaled = np.zeros(n_vars)

    # reconstruire A_scaled complet (p x p) en insérant colonnes supprimées à zéro
    A_scaled = np.zeros((n_vars, n_vars))
    # colonnes gardées -> remplir depuis A_scaled_full
    if n_kept > 0:
        kept_indices = np.where(keep_mask)[0]
        A_scaled[:, kept_indices] = A_scaled_full  # broadcast: rows targets, cols features kept
    # colonnes supprimées restent à zéro

    # remise à l'échelle vers A et b en échelle originale
    # A_{j,k} = (s_k / t_j) * A_scaled_{j,k}
    # b_j = b_scaled_j / t_j
    # construire S diag et T_diag (mais on n'a pas besoin d'inverses matricielles complètes)
    T_diag = t.copy()
    # T_diag protégée précédemment pour éviter 0
    # appliquer remise à l'échelle
    A = np.zeros_like(A_scaled)
    for j in range(n_vars):
        A[j, :] = (s.reshape(1, -1) * A_scaled[j, :]) / T_diag[j]
    b = b_scaled / T_diag

    # si s_k == 0 (col supprimée), s_k*s.. = 0 donc A[:,k] == 0 (déjà garanti ci-dessus)
    # construire residus en échelle originale
    resid = Xn - Xs.dot(A.T) - b.reshape(1, -1)

    # estimer Q (covariance des residus)
    if resid.shape[0] < 2:
        Q = np.diag(np.var(resid, axis=0))
        print("[WARN] Peu d'échantillons pour estimer Q; on utilise diag(var(resid)).")
    else:
        Q = np.cov(resid.T)

    # symétriser + jitter
    Q = (Q + Q.T) / 2.0
    Q = Q + 1e-8 * np.eye(n_vars)

    print("[INFO] A shape:", A.shape, " b shape:", b.shape, " Q shape:", Q.shape)
    return A, b, Q, resid


def select_one_per_day_closest_depth(df, depth_col="depth", target=1.0, tol=1.0):
    if depth_col not in df.columns:
        print(f"[WARN] Colonne de profondeur '{depth_col}' introuvable — on ne sous-échantillonne pas.")
        return df
    if df.shape[0] == 0:
        return df.iloc[0:0]
    tmp = df.copy()
    try:
        idx_naive = tmp.index.tz_convert(None)
    except Exception:
        idx_naive = tmp.index
    tmp = tmp.assign(__date=pd.Index(idx_naive.date))
    picks = []
    for date, g in tmp.groupby("__date"):
        g_valid = g[g[depth_col].notna()]
        if g_valid.empty:
            continue
        absdiff = (g_valid[depth_col] - target).abs()
        min_val = absdiff.min()
        if pd.isna(min_val):
            continue
        if min_val <= tol:
            idx_candidates = absdiff[absdiff == min_val].index
            idxmin = idx_candidates[0]
            picks.append(idxmin)
        else:
            continue
    if len(picks) == 0:
        print("[WARN] Aucun jour n'a de mesure dans la tolérance demandée.")
        return tmp.iloc[0:0].drop(columns="__date")
    result = tmp.loc[picks].drop(columns="__date")
    result = result.sort_index()
    print(f"[INFO] Sous-échantillonnage (tol={tol}m): {len(result)} jours retenus / {len(df)} lignes originales.")
    return result


def enkf_forecast_and_assimilate(A, b, Q, R_diag, x0_mean, x0_cov, observations_df, var_cols,
                                 ens_size=100, assimilate=True, verbose=True):
    n_vars = len(var_cols)
    print(f"[INFO] Lancement EnKF: n_vars={n_vars}, ens_size={ens_size}, assimilate={assimilate}")
    x0_mean = np.asarray(x0_mean).ravel()
    if x0_mean.shape[0] != n_vars:
        print(f"[WARN] Taille de x0_mean attendu {n_vars}, reçu {x0_mean.shape}. Tentative d'ajustement.")
        if x0_mean.size < n_vars:
            new = np.zeros(n_vars)
            new[:x0_mean.size] = x0_mean
            x0_mean = new
        else:
            x0_mean = x0_mean[:n_vars]
    x0_cov = np.asarray(x0_cov)
    if x0_cov.shape != (n_vars, n_vars):
        print(f"[WARN] x0_cov shape attendu {(n_vars,n_vars)}, reçu {x0_cov.shape}. Recalcul avec I*1e-3 fallback.")
        x0_cov = np.eye(n_vars) * 1e-3
    x0_cov = (x0_cov + x0_cov.T) / 2.0
    x0_cov = x0_cov + 1e-8 * np.eye(n_vars)

    R_diag = np.asarray(R_diag).ravel()
    if R_diag.shape[0] != n_vars:
        print(f"[WARN] R_diag length attendu {n_vars}, reçu {R_diag.shape}. On complète avec petites valeurs.")
        newR = np.ones(n_vars) * (np.median(R_diag) if R_diag.size>0 else 1e-6)
        newR[:R_diag.size] = R_diag
        R_diag = newR

    try:
        ens = np.random.multivariate_normal(x0_mean, x0_cov, size=ens_size)
    except Exception as e:
        print("[ERROR] échec tirage multivarié pour initialisation ensemble:", e)
        print("       shapes: mean:", x0_mean.shape, " cov:", x0_cov.shape)
        raise
    ens = ens.T
    print("[INFO] Ensemble initialisé. Moyenne initiale (extrait):", np.round(ens.mean(axis=1)[:5], 4))

    times = observations_df.index
    analyses = []
    forecasts = []
    for t_idx in range(len(times)):
        t = times[t_idx]
        process_noise = np.random.multivariate_normal(np.zeros(n_vars), Q + 1e-12*np.eye(n_vars), size=ens_size).T
        ens = A.dot(ens) + b.reshape(-1,1) + process_noise
        forecasts.append(ens.mean(axis=1).copy())

        if not assimilate:
            analyses.append(ens.mean(axis=1).copy())
            if verbose:
                print(f"[STEP {t_idx+1}/{len(times)}] time={t} (forecast-only)")
            continue

        y_obs = observations_df.iloc[t_idx].values
        obs_mask = ~np.isnan(y_obs)
        n_obs = int(obs_mask.sum())
        if verbose:
            print(f"[STEP {t_idx+1}/{len(times)}] time={t}, obs_count={n_obs}")
        if n_obs == 0:
            analyses.append(ens.mean(axis=1).copy())
            continue
        obs_idx = np.where(obs_mask)[0]
        Ys = ens[obs_idx, :]
        y_mean = Ys.mean(axis=1, keepdims=True)
        x_mean = ens.mean(axis=1, keepdims=True)
        Pf = (ens - x_mean) @ (ens - x_mean).T / (ens_size - 1)
        Pxy = (ens - x_mean) @ (Ys - y_mean).T / (ens_size - 1)
        Pyy = (Ys - y_mean) @ (Ys - y_mean).T / (ens_size - 1)
        R = np.diag(R_diag[obs_idx])
        try:
            inv_term = np.linalg.inv(Pyy + R)
        except np.linalg.LinAlgError:
            inv_term = np.linalg.inv(Pyy + R + 1e-6 * np.eye(n_obs))
        K = Pxy @ inv_term
        y_pert = np.tile(y_obs[obs_idx].reshape(-1,1), (1, ens_size)) + \
                 np.random.multivariate_normal(np.zeros(n_obs), R + 1e-12*np.eye(n_obs), size=ens_size).T
        Ys_pred = Ys
        ens = ens + K @ (y_pert - Ys_pred)
        analyses.append(ens.mean(axis=1).copy())

    df_forecast = pd.DataFrame(np.array(forecasts), index=times, columns=var_cols)
    df_analysis = pd.DataFrame(np.array(analyses), index=times, columns=var_cols)
    return df_forecast, df_analysis


def main():
    print("[START] exécution du script")
    df_all = read_and_clean(PARQUET_PATH)

    # --------- Filtrage géographique robuste et suppression des colonnes lat/lon ----------
    if "longitude" in df_all.columns:
        raw_lon = df_all["longitude"].copy()
        s = raw_lon.astype(str).str.strip().str.replace(",", ".", regex=False)
        s = s.str.extract(r'([+-]?\d+\.?\d*)')[0]
        df_all["longitude"] = pd.to_numeric(s, errors="coerce")

        n_nonnum = df_all["longitude"].isna().sum()
        if n_nonnum > 0:
            print(f"[WARN] {n_nonnum} valeurs de 'longitude' non convertibles -> mises à NaN.")

        before_count = len(df_all)
        # garder longitude >= -69.7 (ou modifie ici pour -71 si tu veux strict)
        mask_keep = df_all["longitude"].notna() & (df_all["longitude"] >= -69.7)
        df_all = df_all.loc[mask_keep].copy()
        removed = before_count - len(df_all)
        print(f"[INFO] Filtrage géographique: {removed} lignes supprimées (longitude < threshold ou non convertible).")
    else:
        print("[WARN] Colonne 'longitude' absente du DataFrame; aucun filtrage géographique appliqué.")

    # Supprimer complètement longitude et latitude pour qu'elles ne soient jamais utilisées
    for col in ["longitude", "latitude", "PRESPR01"]:
        if col in df_all.columns:
            df_all = df_all.drop(columns=[col])
            print(f"[INFO] Colonne '{col}' supprimée du DataFrame (non utilisée pour l'entraînement/affichage).")
    # ---------------------------------------------------------------------------

    df_all, var_cols = apply_qc_rule(df_all)

    if DEPTH_COL in df_all.columns:
        df_all[DEPTH_COL] = pd.to_numeric(df_all[DEPTH_COL], errors='coerce')
        n_non_numeric = df_all[DEPTH_COL].isna().sum()
        print(f"[INFO] Conversion '{DEPTH_COL}' en numérique effectuée. {n_non_numeric} valeurs invalides converties en NaN.")
    else:
        print(f"[WARN] Colonne profondeur '{DEPTH_COL}' introuvable dans le DataFrame.")

    if DEPTH_COL in var_cols:
        var_cols = [c for c in var_cols if c != DEPTH_COL]

    if len(var_cols) == 0:
        raise ValueError("Aucune variable d'état détectée après retrait des colonnes QC/depth.")
    df_all[var_cols] = df_all[var_cols].apply(pd.to_numeric, errors="coerce")
    print(f"[INFO] Variables détectées ({len(var_cols)}): {var_cols[:20]}{'...' if len(var_cols)>20 else ''}")

    # Filtrage par profondeur
    if DEPTH_COL in df_all.columns:
        lower = DEPTH_TARGET - DEPTH_TOL
        upper = DEPTH_TARGET + DEPTH_TOL
        depth_mask = df_all[DEPTH_COL].notna() & (df_all[DEPTH_COL] >= lower) & (df_all[DEPTH_COL] <= upper)
        df_filtered = df_all.loc[depth_mask].copy()
        print(f"[INFO] Après filtrage profondeur ±{DEPTH_TOL}m autour de {DEPTH_TARGET}m : {len(df_filtered)} lignes retenues sur {len(df_all)}.")
    else:
        print(f"[WARN] Colonne profondeur '{DEPTH_COL}' introuvable. Aucun filtrage appliqué.")
        df_filtered = df_all.copy()

    if df_filtered.shape[0] == 0:
        raise ValueError("Aucune ligne retenue après filtrage par profondeur.")

    # split train/test (avant tout resampling)
    train = df_filtered.loc[TRAIN_START:TRAIN_END, var_cols].copy()
    test_original = df_filtered.loc[TEST_START:TEST_END, var_cols].copy()
    print(f"[INFO] Train (après filtrage): {train.shape[0]} lignes, Test original (après filtrage): {test_original.shape[0]} lignes")

    # RESAMPLE DAILY for train; build test daily index but keep only real observations
    agg_method = 'median' if USE_DAILY_MEDIAN else 'mean'
    if RESAMPLE_DAILY:
        print("[INFO] Rééchantillonnage quotidien pour le TRAIN (agrégation journalière: %s ; interpolation) ; TEST: on garde les obs réelles et on crée index journalier pour forecasts" % agg_method)
        train = train.sort_index()
        if train.shape[0] == 0:
            raise ValueError("Le jeu d'entraînement est vide après filtrage; impossible de resampler quotidiennement.")
        # AGRÉGER par jour (mean ou median) puis interpoler
        train_resampled = resample_and_interpolate(train, agg_method=agg_method, freq=AGGREGATION_FREQ)
        train_resampled  = train_resampled .loc[TRAIN_START:TRAIN_END]

        idx_test = pd.date_range(start=TEST_START, end=TEST_END, freq=AGGREGATION_FREQ)
        test_daily = pd.DataFrame(index=idx_test, columns=train_resampled .columns).astype(float)

        if test_original.shape[0] > 0:
            for ts, row in test_original.iterrows():
                try:
                    day = pd.to_datetime(ts).tz_convert(None).normalize()
                except Exception:
                    day = pd.to_datetime(ts).normalize()
                if day in test_daily.index:
                    common_cols = [c for c in row.index if c in test_daily.columns]
                    if len(common_cols) > 0:
                        # on garde les vraies observations (sparse) aux jours correspondants
                        test_daily.loc[day, common_cols] = row[common_cols].values

        train = train_resampled
        test = test_daily
        print(f"[INFO] Train rééchantillonné: {train.shape[0]} jours; Test_daily (obs placées aux jours réels): {test.shape[0]} jours")
    else:
        print("[INFO] RESAMPLE_DAILY == False -> attention aux pas irréguliers")
        test = test_original.copy()

    # add seasonal harmonics & days_since_start
    var_cols_aug = list(var_cols)
    if USE_SEASONAL:
        print(f"[INFO] Ajout des {N_HARMONICS} harmoniques saisonnières")
        train = add_seasonal_harmonics(train, n_harmonics=N_HARMONICS)
        test = add_seasonal_harmonics(test, n_harmonics=N_HARMONICS)
        for k in range(1, N_HARMONICS+1):
            var_cols_aug += [f"sin_{k}", f"cos_{k}"]

    train = add_days_since_start(train)
    test  = add_days_since_start(test)
    if "days_since_start" in train.columns:
        mu = train["days_since_start"].mean()
        std = train["days_since_start"].std() if train["days_since_start"].std() > 0 else 1.0
        train["days_since_start"] = (train["days_since_start"] - mu) / std
        if "days_since_start" in test.columns:
            test["days_since_start"] = (test["days_since_start"] - mu) / std
        var_cols_aug += ["days_since_start"]

    # FILTRER var_cols_aug selon FEATURE_WEIGHTS == 0 pour ignorer complètement
    to_remove = []
    for name in list(var_cols_aug):
        if name in FEATURE_WEIGHTS and float(FEATURE_WEIGHTS[name]) == 0.0:
            to_remove.append(name)
    if len(to_remove) > 0:
        print(f"[INFO] Suppression des colonnes avec FEATURE_WEIGHTS==0 : {to_remove}")
        train = train.drop(columns=[c for c in to_remove if c in train.columns], errors='ignore')
        test  = test.drop(columns=[c for c in to_remove if c in test.columns], errors='ignore')
        var_cols_aug = [v for v in var_cols_aug if v not in to_remove]

    # ensure var_cols_aug valid
    var_cols_aug = [v for v in var_cols_aug if v in train.columns]
    print(f"[INFO] Variables d'état augmentées (len={len(var_cols_aug)}): {var_cols_aug[:30]}{'...' if len(var_cols_aug)>30 else ''}")

    if train.dropna(how='all').shape[0] < 2:
        raise ValueError("Pas assez de données d'entraînement après préparation.")

    # build training pairs on train (daily aggregated + interpolated)
    Xs, Xn = build_training_pairs(train[var_cols_aug])
    # UTILISER la version pondérée
    A, b, Q, resid = estimate_A_weighted(Xs, Xn, var_cols_aug,
                                         feature_weights=FEATURE_WEIGHTS,
                                         target_weights=TARGET_WEIGHTS,
                                         alpha=Ridge_alpha, fit_intercept=USE_INTERCEPT)

    # Diagnostic A & résidus
    eigvals = np.linalg.eigvals(A)
    print("[DIAG] max |eig(A)| =", np.max(np.abs(eigvals)))
    print("[DIAG] valeurs propres (extrait) :", np.round(eigvals[:10], 4))
    print("[DIAG] diag(Q) (extrait) :", np.round(np.diag(Q)[:10], 6))
    print("[DIAG] résidus mean/std (par variable) :", np.round(np.mean(resid, axis=0)[:10], 6),
          np.round(np.std(resid, axis=0)[:10], 6))

    Q = Q * float(Q_scale)
    print("Estimated A shape:", A.shape)

    R_diag = np.var(resid, axis=0) * float(R_scale)
    print("[INFO] R_diag (extrait):", np.round(R_diag[:min(10, len(R_diag))], 6))

    # initial state x0: prefer first non-NaN day in test (daily index) else last train day
    if test.dropna(how='all').shape[0] == 0:
        print("[WARN] Pas d'observations valides dans la période test (après préparation). Utilisation de la dernière ligne du train pour x0.")
        last_train_valid = train.dropna(how='all')
        if last_train_valid.shape[0] == 0:
            raise ValueError("Ni test ni train ne contiennent d'observations valides.")
        first_valid_idx = last_train_valid.index[-1]
        x0_mean = last_train_valid.loc[first_valid_idx].fillna(train.mean()).values
    else:
        first_valid_idx = test.dropna(how='all').index[0]
        x0_mean = test.loc[first_valid_idx].fillna(train.mean()).values

    x0_mean = np.asarray(x0_mean).ravel()
    if x0_mean.shape[0] != len(var_cols_aug):
        print("[WARN] Taille de x0_mean incohérente, on recalcule à partir de la moyenne du train")
        x0_mean = train[var_cols_aug].mean().values

    x0_cov = np.cov(Xs.T) * 0.5 + 1e-6*np.eye(len(var_cols_aug))
    print("[INFO] x0_mean extrait, x0_cov shape:", x0_cov.shape)

    # run EnKF
    test_obs = test[var_cols_aug].copy()
    print("[INFO] Lancement de l'EnKF sur la période test (index journalier)...")
    df_f, df_a = enkf_forecast_and_assimilate(A, b, Q, R_diag, x0_mean, x0_cov, test_obs, var_cols_aug,
                                              ens_size=ENS_SIZE, assimilate=ASSIMILATE)

    # compute metrics using AGGREGATED daily observations (mean or median over each day) -> compare to daily analyses
    test_agg = aggregate_obs(test_original[var_cols], cols=var_cols, agg_method=agg_method, freq=AGGREGATION_FREQ)
    metrics = {}
    for v in var_cols:
        if v not in test_agg.columns or v not in df_a.columns:
            metrics[v] = {"rmse": np.nan, "r2": np.nan}
            continue
        obs_series = test_agg[v].dropna()
        if obs_series.shape[0] == 0:
            metrics[v] = {"rmse": np.nan, "r2": np.nan}
            continue
        # obs_series.index are midnight datetimes -> intersect with df_a.index
        common_idx = obs_series.index.intersection(df_a.index)
        if len(common_idx) == 0:
            metrics[v] = {"rmse": np.nan, "r2": np.nan}
            continue
        y_true = obs_series.loc[common_idx].values
        y_pred = df_a.loc[common_idx, v].values
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = np.nan
        metrics[v] = {"rmse": rmse, "r2": r2}

    print("[INFO] Metrics calculés pour chaque variable (extrait):")
    for k in list(metrics.keys())[:10]:
        print(f" {k}: RMSE={metrics[k]['rmse']:.4g}, R2={metrics[k]['r2']:.4g}")

    # debug
    print("\n[DEBUG] extrait A (top-left 5x5):")
    print(np.round(A[:5, :5], 6))
    print("[DEBUG] extrait b (premières valeurs):", np.round(b[:5], 6))
    print("[DEBUG] diag(Q) (premières valeurs):", np.round(np.diag(Q)[:10], 8))
    print("[DEBUG] diag(R_diag) (premières valeurs):", np.round(R_diag[:10], 8))

    # Save plots
    print(f"[INFO] Sauvegarde des graphiques et diagnostics dans le PDF: {OUTPUT_PDF}")
    with PdfPages(OUTPUT_PDF) as pdf:
        for v in var_cols:
            fig, ax = plt.subplots(figsize=(10, 4))
            # afficher les observations journalières agrégées (mean/median)
            if v in test_agg.columns and (~test_agg[v].isna()).any():
                ax.plot(test_agg.index, test_agg[v], marker='o', linestyle='none', label=f"obs_{agg_method} {v}")
            # afficher analyses quotidiennes
            if v in df_a.columns:
                ax.plot(df_a.index, df_a[v], label=f"analysis {v}")
            ax.legend(loc="upper right")
            ax.set_title(f"{v} (profondeur ∈ [{DEPTH_TARGET-DEPTH_TOL}, {DEPTH_TARGET+DEPTH_TOL}] m) — forecast " + AGGREGATION_FREQ)
            ax.set_xlabel("time")
            ax.set_ylabel(v)
            rmse = metrics[v]["rmse"]
            r2 = metrics[v]["r2"]
            rmse_str = f"{rmse:.4g}" if not np.isnan(rmse) else "n/a"
            r2_str = f"{r2:.4g}" if not np.isnan(r2) else "n/a"
            metrics_text = f"R² = {r2_str}    RMSE = {rmse_str}    (obs_agg={agg_method})"
            fig.text(0.1, 0.02, metrics_text, ha='left', fontsize=10)
            fig.tight_layout(rect=[0, 0.04, 1, 1])
            pdf.savefig(fig)
            plt.close(fig)

        # summary page
        try:
            summary_df = pd.DataFrame.from_dict({k: {"rmse": metrics[k]["rmse"], "r2": metrics[k]["r2"]} for k in var_cols}, orient="index")
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.set_title("Résumé des metrics par variable (données filtrées par profondeur)")
            table = ax.table(cellText=np.round(summary_df.fillna(np.nan), 6).astype(object).values,
                             colLabels=summary_df.columns,
                             rowLabels=summary_df.index,
                             loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            print("[WARN] Impossible d'ajouter la page de résumé:", e)

        # diagnostics page (params, diag Q/R, eig(A))
        try:
            params = {
                "ENS_SIZE": ENS_SIZE,
                "RESAMPLE_DAILY": RESAMPLE_DAILY,
                "USE_SEASONAL": USE_SEASONAL,
                "N_HARMONICS": N_HARMONICS,
                "USE_INTERCEPT": USE_INTERCEPT,
                "ASSIMILATE": ASSIMILATE,
                "Q_scale": Q_scale,
                "R_scale": R_scale,
                "Ridge_alpha": Ridge_alpha,
                "SEED": SEED,
                "TRAIN_PERIOD": f"{TRAIN_START} -> {TRAIN_END}",
                "TEST_PERIOD": f"{TEST_START} -> {TEST_END}",
                "DEPTH_TARGET": DEPTH_TARGET,
                "DEPTH_TOL": DEPTH_TOL,
                "AGG_METHOD": agg_method,
            }
            params_lines = [f"{k}: {v}" for k, v in params.items()]
            params_text = "\n".join(params_lines)

            diagQ = np.round(np.diag(Q), 6)
            Rvals = np.round(R_diag, 6)
            nshow_diag = min(20, len(diagQ))
            diagQ_text = ", ".join(map(str, diagQ[:nshow_diag]))
            Rvals_text = ", ".join(map(str, Rvals[:nshow_diag]))

            try:
                eigvals = np.linalg.eigvals(A)
                nshow_eig = min(10, len(eigvals))
                eig_text = ", ".join([f"{v:.4f}" for v in eigvals[:nshow_eig]])
                max_eig = np.max(np.abs(eigvals))
            except Exception:
                eig_text = "n/a"
                max_eig = np.nan

            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.set_title("Paramètres et diagnostics (Q, R, valeurs propres de A)")

            x0 = 0.05
            y = 0.92
            ax.text(x0, y, "Hyper-paramètres :", transform=fig.transFigure, fontsize=11, weight='bold', va='top')
            ax.text(x0, y-0.035, params_text, transform=fig.transFigure, fontsize=9, va='top')

            ax.text(x0, y-0.34, "Extrait diag(Q) (premières valeurs) :", transform=fig.transFigure, fontsize=11, weight='bold', va='top')
            ax.text(x0, y-0.37, diagQ_text, transform=fig.transFigure, fontsize=8, va='top')

            ax.text(x0, y-0.46, "Extrait R_diag (premières valeurs) :", transform=fig.transFigure, fontsize=11, weight='bold', va='top')
            ax.text(x0, y-0.49, Rvals_text, transform=fig.transFigure, fontsize=8, va='top')

            ax.text(x0, y-0.58, "Valeurs propres de A (extrait) :", transform=fig.transFigure, fontsize=11, weight='bold', va='top')
            ax.text(x0, y-0.61, eig_text, transform=fig.transFigure, fontsize=9, va='top')
            ax.text(x0, y-0.67, f"max |eig(A)| = {max_eig:.4f}", transform=fig.transFigure, fontsize=9, va='top')

            try:
                cols_display = var_cols_aug[:nshow_diag]
                names_text = ", ".join(cols_display)
                ax.text(x0, y-0.74, f"Correspondance variables (premières {nshow_diag}):", transform=fig.transFigure, fontsize=10, weight='bold', va='top')
                ax.text(x0, y-0.77, names_text, transform=fig.transFigure, fontsize=8, va='top')
            except Exception:
                pass

            ax.text(0.05, 0.05, "NB: diag(Q) et R_diag sont les variances de bruit de processus / observation.\nAjuste Q_scale et R_scale pour affiner l'assimilation/forecast.", transform=fig.transFigure, fontsize=8)

            pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            print("[WARN] Impossible d'ajouter la page de diagnostics Q/R au PDF:", e)

    print(f"[INFO] PDF sauvegardé: {OUTPUT_PDF}")
    print("[END] exécution terminée")
    return {"A": A, "b": b, "Q": Q, "R_diag": R_diag, "forecast": df_f, "analysis": df_a, "metrics": metrics, "var_cols_aug": var_cols_aug}

if __name__ == "__main__":
    out = main()
