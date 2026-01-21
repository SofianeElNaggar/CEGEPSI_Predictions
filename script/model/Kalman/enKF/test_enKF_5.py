import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------------------------------------------------------
# Configuration générale et paramètres
# Chaque variable suivante configure le comportement du pipeline (lecture, filtre,
# préparation, entraînement, assimilation, sortie).
# -----------------------------------------------------------------------------

# colonne contenant la profondeur dans le dataset
DEPTH_COL = "depth"
# profondeur cible (en mètres) autour de laquelle on souhaite sélectionner les profils/mesures
DEPTH_TARGET = 1
# tolérance autour de DEPTH_TARGET (± en m)
DEPTH_TOL = 0.1

# Flags et paramètres d'ingénierie des features
USE_SEASONAL = True      # ajouter ou non des harmoniques saisonnières (sin/cos)
N_HARMONICS = 3          # nombre d'harmoniques saisonnières à ajouter
USE_INTERCEPT = True     # inclure un terme d'interception pour la régression
ASSIMILATE = False       # si True on assimile les observations dans l'EnKF
Q_scale = 1.0            # facteur multiplicatif sur Q estimée (bruit de processus)
R_scale = 1.0            # facteur multiplicatif sur R estimée (bruit d'observation)
RESAMPLE_DAILY = True    # ré-échantillonner le train quotidiennement

# ---------- CONFIG fichier / période d'entraînement & test -------------------
PARQUET_PATH = "../../dataset/OMDS-CTD datalight_with_pos.parquet"
TIME_COL = "time"
TRAIN_START = "2000-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2025-12-30"
ENS_SIZE = 100
Ridge_alpha = 1.0
SEED = 42

# variables à prédire
PREDICT_VARS = ["TEMPS901"]

OUTPUT_PDF = "../results/prediction/enKF/v5/predictions_report_Q=" + str(Q_scale) + "_" + "_".join(PREDICT_VARS) + ".pdf"

# reproducibilité
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Poids utilisés pour l'estimation pondérée (estimate_A_weighted)
# FEATURE_WEIGHTS : poids appliqués aux colonnes explicatives (features)
# TARGET_WEIGHTS  : poids appliqués aux colonnes cibles (targets)
# Des poids plus grands renforcent l'importance de la variable dans la
# régression; 0 signifie complètement ignorer.
# -----------------------------------------------------------------------------
FEATURE_WEIGHTS = {
    "TEMPS901": 1.0,
    "CPHLPR01": 1.0,
    "TURBPR01": 1.0,
    "PHXXPR01": 1.0,
    "PSALST01": 1.0,
    "SIGTEQST": 1.0,
    "DOXYZZ01": 1.0,
    "sin_1": 2.5, "cos_1": 2.5,
    "sin_2": 1.0, "cos_2": 1.0,
    "sin_3": 0.5, "cos_3": 0.5,
    "days_since_start": 2.0,
}
TARGET_WEIGHTS = {
    "TEMPS901": 1.0,
    "CPHLPR01": 1.0,
    "TURBPR01": 1.0,
    "PHXXPR01": 1.0,
    "PSALST01": 1.0,
    "SIGTEQST": 1.0,
    "DOXYZZ01": 1.0
}

# -----------------------------------------------------------------------------
# read_and_clean(path)
# Lecture du parquet, conversion de la colonne temporelle et mise en index
# - path: chemin vers le fichier parquet
# Retour: DataFrame pandas indexé par la colonne TIME_COL
# Lève une erreur si la colonne temporelle est introuvable
# -----------------------------------------------------------------------------

def read_and_clean(path):
    import time
    t0 = time.time()
    print("[INFO] Lecture du fichier parquet :", path)
    df = pd.read_parquet(path)

    print(f"[INFO] Lecture terminée en {time.time() - t0:.2f}s, {len(df):,} lignes, {len(df.columns)} colonnes")

    # Certains fichiers peuvent avoir la colonne UTC. On la renomme vers TIME_COL
    if "UTC" in df.columns and TIME_COL not in df.columns:
        df = df.rename(columns={"UTC": TIME_COL})

    if TIME_COL not in df.columns:
        # arrêt explicite si aucune colonne temporelle trouvée
        raise KeyError(f"Colonne temporelle '{TIME_COL}' introuvable dans le fichier parquet.")

    # S'assurer que la colonne temporelle est datetime (avec gestion des erreurs)
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        print("[INFO] Conversion de la colonne time en datetime (format initial =", df[TIME_COL].dtype, ")")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
        n_bad = df[TIME_COL].isna().sum()
        if n_bad > 0:
            print(f"[INFO] Suppression de {n_bad} lignes avec time non parseable.")
            # suppression des lignes dont la date n'a pas pu être parsée
            df = df.dropna(subset=[TIME_COL])

    # mettre la colonne temporelle en index pour faciliter les opérations temporelles
    df = df.set_index(TIME_COL).sort_index()
    print(f"[INFO] Nettoyage terminé en {time.time() - t0:.2f}s total")
    return df


# -----------------------------------------------------------------------------
# apply_qc_rule(df)
# Applique un masque de qualité (QC) pour mettre à NaN les valeurs marquées
# comme invalides. Les colonnes de QC doivent se terminer par "_QC".
# - df: DataFrame d'entrée
# Retour: (df nettoyé, var_cols) où var_cols sont les colonnes non-QC
# -----------------------------------------------------------------------------
def apply_qc_rule(df):
    print("[INFO] Application des règles QC")
    # détecter les colonnes de QC (suffixe '_QC')
    qc_cols = [c for c in df.columns if c.endswith("_QC")]
    # colonnes de variables réelles = toutes les colonnes sauf les colonnes QC
    var_cols = [c for c in df.columns if c not in qc_cols]
    print(f"[INFO] Colonnes QC détectées: {qc_cols}")

    # pour chaque colonne QC, si la valeur du QC n'est ni 1 ni 5 on considère la
    # donnée correspondante comme invalide et on la remplace par NaN
    for qc in qc_cols:
        var = qc[:-3]
        if var in df.columns:
            # mask_bad True pour les valeurs QC non nulles et différentes de 1 ou 5
            mask_bad = df[qc].notna() & ((df[qc] != 1) & (df[qc] != 5))
            n_bad = mask_bad.sum()
            if n_bad > 0:
                print(f"[INFO] QC: {n_bad} valeurs marquées comme invalides pour '{var}' (colonne {qc}).")
                df.loc[mask_bad, var] = np.nan
    return df, var_cols


# -----------------------------------------------------------------------------
# resample_daily_and_interpolate(df)
# Rééchantillonne le DataFrame en fréquence quotidienne (mean par jour) puis
# interpole les valeurs manquantes dans le temps.
# - gère les index timezone-aware en les convertissant en naive
# - retourne un DataFrame avec index journalier
# -----------------------------------------------------------------------------
def resample_daily_and_interpolate(df):
    tmp = df.copy()
    # conversion de l'index timezone-aware -> timezone-naive pour resampling
    try:
        tmp.index = tmp.index.tz_convert(None)
    except Exception:
        pass
    # moyenne par jour
    df_daily = tmp.resample("D").mean()
    # interpolation temporelle (utile si plusieurs mesures par jour ou trous courts)
    df_daily = df_daily.interpolate(method="time", limit_direction="both")
    return df_daily


# -----------------------------------------------------------------------------
# add_seasonal_harmonics(df, n_harmonics=3)
# Ajoute des colonnes sin_k / cos_k pour capturer la saisonnalité annuelle
# - utilité: fournir des features périodiques à la régression/EnKF
# - Le calcul est basé sur dayofyear / 365.25
# -----------------------------------------------------------------------------
def add_seasonal_harmonics(df, n_harmonics=3):
    tmp = df.copy()
    try:
        idx_naive = tmp.index.tz_convert(None)
    except Exception:
        idx_naive = tmp.index
    # jour de l'année (1-366)
    doy = idx_naive.dayofyear.values.astype(float)
    for k in range(1, n_harmonics+1):
        ang = 2.0 * np.pi * k * (doy / 365.25)
        tmp[f"sin_{k}"] = np.sin(ang)
        tmp[f"cos_{k}"] = np.cos(ang)
    return tmp


# -----------------------------------------------------------------------------
# add_days_since_start(df)
# Ajoute la feature 'days_since_start' calculée par rapport à la première date
# du DataFrame (en jours). Si l'index est vide renvoie NaN.
# -----------------------------------------------------------------------------
def add_days_since_start(df):
    tmp = df.copy()
    try:
        idx_naive = tmp.index.tz_convert(None)
    except Exception:
        idx_naive = tmp.index
    if len(idx_naive) == 0:
        tmp["days_since_start"] = np.nan
        return tmp
    # différence en secondes puis conversion en jours
    days = (idx_naive - idx_naive[0]).total_seconds() / (24*3600.0)
    tmp["days_since_start"] = days
    return tmp


# -----------------------------------------------------------------------------
# build_training_pairs(X)
# Construit des paires (x_t, x_{t+1}) nécessaires pour estimer la matrice A
# - X: DataFrame chronologique des variables d'état
# - Ignore toute paire contenant des NaN
# Retour: deux arrays numpy Xs, Xn où chaque ligne correspond à x_t / x_{t+1}
# Lève ValueError si aucune paire valide n'est trouvée.
# -----------------------------------------------------------------------------
def build_training_pairs(X):
    print("[INFO] Construction des paires d'entraînement (x_t, x_{t+1})")
    Xs = []
    Xn = []
    # itérer sur la séquence et garder les paires consécutives valides
    for i in range(len(X)-1):
        xt = X.iloc[i].values
        xt1 = X.iloc[i+1].values
        # si l'une des deux contient un NaN, on saute la paire
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


# -----------------------------------------------------------------------------
# estimate_A_weighted(...)
# Estime la matrice A et le biais b d'un modèle linéaire X_{t+1} = A X_t + b
# - Utilise sklearn Ridge (regularisation L2) pour stabiliser l'estimation
# - Implémente un schéma de pondération: on multiplie les colonnes (features)
#   et les targets par des poids afin d'accorder plus/moins d'importance
# - Calcule également la covariance des résidus Q (bruit de processus)
# Paramètres principaux:
#  Xs, Xn : arrays (n_samples, n_vars) avec x_t / x_{t+1}
#  var_cols: liste des noms de variables (utilisée pour associer les poids)
#  feature_weights / target_weights: dict name -> weight
#  alpha: paramètre ridge
#  fit_intercept: bool
# Retour: A, b, Q, resid
# -----------------------------------------------------------------------------
def estimate_A_weighted(Xs, Xn, var_cols, feature_weights=None, target_weights=None, alpha=1.0, fit_intercept=True):
    print("[INFO] Estimation de A pondéré (fit_intercept=%s)" % fit_intercept)
    n_vars = Xs.shape[1]
    if len(var_cols) != n_vars:
        # cohérence entre la forme des arrays et les noms attendus
        raise ValueError("var_cols length inconsistent with Xs shape.")

    feature_weights = feature_weights or {}
    target_weights = target_weights or {}

    # s = poids appliqué aux colonnes explicatives (features)
    # t = poids appliqué aux colonnes cibles (targets)
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

    # éviter zéros exacts qui poseraient problème dans les inversions
    s[s == 0.0] = 1e-8
    t[t == 0.0] = 1e-8

    # application des poids (mise à l'échelle simple par multiplication)
    Xs_scaled = Xs * s.reshape(1, -1)
    Xn_scaled = Xn * t.reshape(1, -1)

    A_scaled = np.zeros((n_vars, n_vars))
    b_scaled = np.zeros(n_vars)
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)

    # Estimer une régression par colonne de target (multi-sortie fait manuellement)
    for j in range(n_vars):
        y = Xn_scaled[:, j]
        model.fit(Xs_scaled, y)
        A_scaled[j, :] = model.coef_
        if fit_intercept:
            b_scaled[j] = model.intercept_

    # ramenons les coefficients à l'échelle originale
    S = np.diag(s)
    T = np.diag(t)
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        # fallback si T singulier
        T_inv = np.diag(1.0 / np.where(np.diag(T) == 0.0, 1e-8, np.diag(T)))
    A_original_T = S.dot(A_scaled.T.dot(T_inv))
    A = A_original_T.T
    b = b_scaled / np.diag(T)

    # résidus empiriques: résidu = Xn - (Xs A^T + b)
    resid = Xn - Xs.dot(A.T) - b.reshape(1, -1)

    # Estimation de Q: covariance des résidus. Si peu d'échantillons on prend
    # simplement la diagonale des variances comme approximation.
    if resid.shape[0] < 2:
        Q = np.diag(np.var(resid, axis=0))
        print("[WARN] Peu d'échantillons pour estimer Q; on utilise diag(var(resid)).")
    else:
        Q = np.cov(resid.T)
    # symétriser numériquement et ajouter petite valeur pour stabilité
    Q = (Q + Q.T) / 2.0
    Q = Q + 1e-8 * np.eye(n_vars)

    print("[INFO] A shape:", A.shape, " b shape:", b.shape, " Q shape:", Q.shape)
    return A, b, Q, resid


# -----------------------------------------------------------------------------
# select_one_per_day_closest_depth(...)
# Pour chaque jour, choisir l'échantillon (ligne) dont la profondeur est la
# plus proche de la profondeur de cible `target`, à condition d'être dans la
# tolérance `tol`.
# - Si la colonne depth est absente renvoie le DataFrame inchangé
# - Retourne subset indexé par les timestamps originaux (ordonnés)
# -----------------------------------------------------------------------------
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
    # créer une colonne temporaire avec la date (sans l'heure) pour grouper
    tmp = tmp.assign(__date=pd.Index(idx_naive.date))
    picks = []
    # parcourir chaque jour et choisir le point dont depth est le plus proche de target
    for date, g in tmp.groupby("__date"):
        g_valid = g[g[depth_col].notna()]
        if g_valid.empty:
            continue
        absdiff = (g_valid[depth_col] - target).abs()
        min_val = absdiff.min()
        if pd.isna(min_val):
            continue
        if min_val <= tol:
            # si plusieurs candidats à égalité, on prend le premier
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


# -----------------------------------------------------------------------------
# enkf_forecast_and_assimilate(...)
# Implémentation basique de l'Ensemble Kalman Filter (EnKF) en mode entrées/sorties
# - A, b: modèle linéaire d'évolution d'état x_{t+1} = A x_t + b + w (w ~ N(0,Q))
# - Q: covariance du bruit de processus
# - R_diag: vecteur des variances d'observation (diagonale de R)
# - x0_mean, x0_cov: moyenne et covariance initiales pour l'ensemble
# - observations_df: DataFrame indexé par temps contenant observations (NaN=absence)
# - var_cols: noms des variables d'état (ordre important)
# - ens_size: taille de l'ensemble
# - assimilate: si False -> forecast-only (pas de correction par observations)
# Retour: (df_forecast, df_analysis) DataFrames indexés par times
# -----------------------------------------------------------------------------
def enkf_forecast_and_assimilate(A, b, Q, R_diag, x0_mean, x0_cov, observations_df, var_cols, ens_size=100, assimilate=True, verbose=True):
    n_vars = len(var_cols)
    print(f"[INFO] Lancement EnKF: n_vars={n_vars}, ens_size={ens_size}, assimilate={assimilate}")

    # mise en forme et vérifications de sécurité
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
        # fallback: covariance identique (petite valeur) si forme incorrecte
        print(f"[WARN] x0_cov shape attendu {(n_vars,n_vars)}, reçu {x0_cov.shape}. Recalcul avec I*1e-3 fallback.")
        x0_cov = np.eye(n_vars) * 1e-3
    # symétriser et régulariser numériquement
    x0_cov = (x0_cov + x0_cov.T) / 2.0
    x0_cov = x0_cov + 1e-8 * np.eye(n_vars)

    # R_diag doit être de longueur n_vars; sinon on complète
    R_diag = np.asarray(R_diag).ravel()
    if R_diag.shape[0] != n_vars:
        print(f"[WARN] R_diag length attendu {n_vars}, reçu {R_diag.shape}. On complète avec petites valeurs.")
        newR = np.ones(n_vars) * (np.median(R_diag) if R_diag.size>0 else 1e-6)
        newR[:R_diag.size] = R_diag
        R_diag = newR

    # initialisation de l'ensemble (tirage multivarié)
    try:
        ens = np.random.multivariate_normal(x0_mean, x0_cov, size=ens_size)
    except Exception as e:
        print("[ERROR] échec tirage multivarié pour initialisation ensemble:", e)
        print("       shapes: mean:", x0_mean.shape, " cov:", x0_cov.shape)
        raise
    # on arrange ens de shape (n_vars, ens_size) pour simplifier les opérations mat.
    ens = ens.T
    print("[INFO] Ensemble initialisé. Moyenne initiale (extrait):", np.round(ens.mean(axis=1)[:5], 4))

    times = observations_df.index
    analyses = []
    forecasts = []

    # boucle temporelle: 1 step prédiction + assimilation (si activée)
    for t_idx in range(len(times)):
        t = times[t_idx]
        # tirage bruit de processus pour chaque membre
        process_noise = np.random.multivariate_normal(np.zeros(n_vars), Q + 1e-12*np.eye(n_vars), size=ens_size).T
        ens = A.dot(ens) + b.reshape(-1,1) + process_noise
        forecasts.append(ens.mean(axis=1).copy())

        if not assimilate:
            # mode forecast-only (pas de mise à jour avec les observations)
            analyses.append(ens.mean(axis=1).copy())
            if verbose:
                print(f"[STEP {t_idx+1}/{len(times)}] time={t} (forecast-only)")
            continue

        # récupérer l'observation y (NaN si manquante)
        y_obs = observations_df.iloc[t_idx].values
        obs_mask = ~np.isnan(y_obs)
        n_obs = int(obs_mask.sum())
        if verbose:
            print(f"[STEP {t_idx+1}/{len(times)}] time={t}, obs_count={n_obs}")
        if n_obs == 0:
            analyses.append(ens.mean(axis=1).copy())
            continue

        # indices d'observation et sous-ensembles correspondants
        obs_idx = np.where(obs_mask)[0]
        Ys = ens[obs_idx, :]
        y_mean = Ys.mean(axis=1, keepdims=True)
        x_mean = ens.mean(axis=1, keepdims=True)

        # calcul des covariances empiriques Pf, Pxy, Pyy
        Pf = (ens - x_mean) @ (ens - x_mean).T / (ens_size - 1)
        Pxy = (ens - x_mean) @ (Ys - y_mean).T / (ens_size - 1)
        Pyy = (Ys - y_mean) @ (Ys - y_mean).T / (ens_size - 1)

        # matrice R pour les observations présentes (diagonale)
        R = np.diag(R_diag[obs_idx])
        try:
            inv_term = np.linalg.inv(Pyy + R)
        except np.linalg.LinAlgError:
            # régularisation si Pyy+R singulier
            inv_term = np.linalg.inv(Pyy + R + 1e-6 * np.eye(n_obs))
        # gain de Kalman
        K = Pxy @ inv_term

        # perturbation stochastique des observations pour EnKF stochastique
        y_pert = np.tile(y_obs[obs_idx].reshape(-1,1), (1, ens_size)) + \
                 np.random.multivariate_normal(np.zeros(n_obs), R + 1e-12*np.eye(n_obs), size=ens_size).T
        Ys_pred = Ys
        # mise-à-jour de chaque membre de l'ensemble
        ens = ens + K @ (y_pert - Ys_pred)
        analyses.append(ens.mean(axis=1).copy())

    # mettre sous forme DataFrame pour sortie
    df_forecast = pd.DataFrame(np.array(forecasts), index=times, columns=var_cols)
    df_analysis = pd.DataFrame(np.array(analyses), index=times, columns=var_cols)
    return df_forecast, df_analysis


# -----------------------------------------------------------------------------
# main()
# Orchestration complète du pipeline : lecture, QC, filtrage profondeur,
# préparation (rééchantillonnage, features saisonnières, normalisation),
# estimation A/b/Q, et exécution de l'EnKF puis sauvegarde des diagnostics au PDF.
# Retour: dictionnaire contenant A, b, Q, R_diag, forecast, analysis, metrics...
# -----------------------------------------------------------------------------
def main():
    print("[START] exécution du script")
    df_all = read_and_clean(PARQUET_PATH)

    # --------- Filtrage géographique robuste et suppression des colonnes lat/lon ----------
    if "longitude" in df_all.columns:
        raw_lon = df_all["longitude"].copy()
        # certaines valeurs texte peuvent contenir des virgules ou d'autres caractères
        s = raw_lon.astype(str).str.strip().str.replace(",", ".", regex=False)
        s = s.str.extract(r'([+-]?\d+\.?\d*)')[0]
        df_all["longitude"] = pd.to_numeric(s, errors="coerce")

        n_nonnum = df_all["longitude"].isna().sum()
        if n_nonnum > 0:
            print(f"[WARN] {n_nonnum} valeurs de 'longitude' non convertibles -> mises à NaN.")

        before_count = len(df_all)
        # seuil arbitraire pour filtrer la zone géographique d'intérêt
        mask_keep = df_all["longitude"].notna() & (df_all["longitude"] >= -69.7)
        df_all = df_all.loc[mask_keep].copy()
        removed = before_count - len(df_all)
        print(f"[INFO] Filtrage géographique: {removed} lignes supprimées (longitude < threshold ou non convertible).")
    else:
        print("[WARN] Colonne 'longitude' absente du DataFrame; aucun filtrage géographique appliqué.")

    # suppression des colonnes non utilisées
    for col in ["longitude", "latitude", "PRESPR01"]:
        if col in df_all.columns:
            df_all = df_all.drop(columns=[col])
            print(f"[INFO] Colonne '{col}' supprimée du DataFrame (non utilisée pour l'entraînement/affichage).")
    # ---------------------------------------------------------------------------

    # appliquer règles QC et récupérer la liste des colonnes de variables
    df_all, var_cols = apply_qc_rule(df_all)

    # convertir la colonne profondeur en numérique si présente
    if DEPTH_COL in df_all.columns:
        df_all[DEPTH_COL] = pd.to_numeric(df_all[DEPTH_COL], errors='coerce')
        n_non_numeric = df_all[DEPTH_COL].isna().sum()
        print(f"[INFO] Conversion '{DEPTH_COL}' en numérique effectuée. {n_non_numeric} valeurs invalides converties en NaN.")
    else:
        print(f"[WARN] Colonne profondeur '{DEPTH_COL}' introuvable dans le DataFrame.")

    # si depth est listée parmi var_cols on la retire (on ne veut pas l'inclure dans l'état dynamique)
    if DEPTH_COL in var_cols:
        var_cols = [c for c in var_cols if c != DEPTH_COL]

    if len(var_cols) == 0:
        raise ValueError("Aucune variable d'état détectée après retrait des colonnes QC/depth.")
    # forcer conversion numeric pour les variables d'état
    df_all[var_cols] = df_all[var_cols].apply(pd.to_numeric, errors="coerce")
    print(f"[INFO] Variables détectées ({len(var_cols)}): {var_cols[:20]}{'...' if len(var_cols)>20 else ''}")

    # Filtrage par profondeur autour de DEPTH_TARGET
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

    # RESAMPLE DAILY pour l'entraînement; créer un index de test quotidien mais ne conserver que les observations réelles
    if RESAMPLE_DAILY:
        print("[INFO] Rééchantillonnage quotidien pour le TRAIN (interpolation) ; TEST: on garde les obs réelles et on crée index journalier pour forecasts")
        train = train.sort_index()
        if train.shape[0] == 0:
            raise ValueError("Le jeu d'entraînement est vide après filtrage; impossible de resampler quotidiennement.")
        train_daily = resample_daily_and_interpolate(train)
        train_daily = train_daily.loc[TRAIN_START:TRAIN_END]

        # construire un index quotidien pour la période de test
        idx_test = pd.date_range(start=TEST_START, end=TEST_END, freq='D')
        test_daily = pd.DataFrame(index=idx_test, columns=train_daily.columns).astype(float)

        # placer les observations réelles du test_original sur les jours correspondants
        if test_original.shape[0] > 0:
            for ts, row in test_original.iterrows():
                try:
                    day = pd.to_datetime(ts).tz_convert(None).normalize()
                except Exception:
                    day = pd.to_datetime(ts).normalize()
                if day in test_daily.index:
                    common_cols = [c for c in row.index if c in test_daily.columns]
                    if len(common_cols) > 0:
                        test_daily.loc[day, common_cols] = row[common_cols].values

        train = train_daily
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

    # ajout de la feature trend 'days_since_start' et normalisation (z-score)
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

    # s'assurer que var_cols_aug est contenu dans train
    var_cols_aug = [v for v in var_cols_aug if v in train.columns]
    print(f"[INFO] Variables d'état augmentées (len={len(var_cols_aug)}): {var_cols_aug[:30]}{'...' if len(var_cols_aug)>30 else ''}")

    if train.dropna(how='all').shape[0] < 2:
        raise ValueError("Pas assez de données d'entraînement après préparation.")

    # build training pairs on train (daily interpolated)
    Xs, Xn = build_training_pairs(train[var_cols_aug])

    # UTILISER la version pondérée pour estimer A, b, Q
    A, b, Q, resid = estimate_A_weighted(Xs, Xn, var_cols_aug,
                                         feature_weights=FEATURE_WEIGHTS,
                                         target_weights=TARGET_WEIGHTS,
                                         alpha=Ridge_alpha, fit_intercept=USE_INTERCEPT)

    # Diagnostics simples: valeurs propres, diag(Q), stats des résidus
    eigvals = np.linalg.eigvals(A)
    print("[DIAG] max |eig(A)| =", np.max(np.abs(eigvals)))
    print("[DIAG] valeurs propres (extrait) :", np.round(eigvals[:10], 4))
    print("[DIAG] diag(Q) (extrait) :", np.round(np.diag(Q)[:10], 6))
    print("[DIAG] résidus mean/std (par variable) :", np.round(np.mean(resid, axis=0)[:10], 6), np.round(np.std(resid, axis=0)[:10], 6))

    # ajuster Q et R selon les échelles fournies
    Q = Q * float(Q_scale)
    print("Estimated A shape:", A.shape)

    R_diag = np.var(resid, axis=0) * float(R_scale)
    print("[INFO] R_diag (extrait):", np.round(R_diag[:min(10, len(R_diag))], 6))

    # initial state x0 : on choisit la première observation valide du test si disponible sinon la dernière du train
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

    # covariance initiale x0_cov basée sur Xs (variabilité historique)
    x0_cov = np.cov(Xs.T) * 0.5 + 1e-6*np.eye(len(var_cols_aug))
    print("[INFO] x0_mean extrait, x0_cov shape:", x0_cov.shape)

    # run EnKF
    test_obs = test[var_cols_aug].copy()
    print("[INFO] Lancement de l'EnKF sur la période test (index journalier)...")
    df_f, df_a = enkf_forecast_and_assimilate(A, b, Q, R_diag, x0_mean, x0_cov, test_obs, var_cols_aug,
                                              ens_size=ENS_SIZE, assimilate=ASSIMILATE)

    # calculer les métriques à l'aide des observations ORIGINALES (test_original) MAIS seulement pour PREDICT_VARS
    metrics = {}
    for v in PREDICT_VARS:
        if v not in test_original.columns or v not in df_a.columns:
            metrics[v] = {"rmse": np.nan, "r2": np.nan}
            continue
        obs_dates = []
        obs_values = []
        # collecter les observations réelles et normaliser la date en jour
        for ts, row in test_original[v].dropna().items():
            try:
                day = pd.to_datetime(ts).tz_convert(None).normalize()
            except Exception:
                day = pd.to_datetime(ts).normalize()
            obs_dates.append(day)
            obs_values.append(row)
        if len(obs_dates) == 0:
            metrics[v] = {"rmse": np.nan, "r2": np.nan}
            continue
        obs_idx = pd.DatetimeIndex(obs_dates)
        mask_in = obs_idx.isin(df_a.index)
        if mask_in.sum() == 0:
            metrics[v] = {"rmse": np.nan, "r2": np.nan}
            continue
        obs_idx_in = obs_idx[mask_in]
        y_true = np.array([obs_values[i] for i in range(len(obs_dates)) if mask_in[i]])
        y_pred = df_a.loc[obs_idx_in, v].values
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = np.nan
        metrics[v] = {"rmse": rmse, "r2": r2}

    print("[INFO] Metrics calculés pour les variables demandées :")
    for k in PREDICT_VARS:
        print(f" {k}: RMSE={metrics[k]['rmse']:.4g}, R2={metrics[k]['r2']:.4g}")

    # debug prints pour inspection rapide
    print("\n[DEBUG] extrait A (top-left 5x5):")
    print(np.round(A[:5, :5], 6))
    print("[DEBUG] extrait b (premières valeurs):", np.round(b[:5], 6))
    print("[DEBUG] diag(Q) (premières valeurs):", np.round(np.diag(Q)[:10], 8))
    print("[DEBUG] diag(R_diag) (premières valeurs):", np.round(R_diag[:10], 8))

    # Save plots only for PREDICT_VARS
    print(f"[INFO] Sauvegarde des graphiques et diagnostics dans le PDF: {OUTPUT_PDF}")
    with PdfPages(OUTPUT_PDF) as pdf:
        for v in PREDICT_VARS:
            fig, ax = plt.subplots(figsize=(10, 4))
            if (~test_original[v].isna()).any():
                ax.plot(test_original.index, test_original[v], marker='.', linestyle='none', label=f"obs {v}")
            if v in df_a.columns:
                ax.plot(df_a.index, df_a[v], label=f"analysis {v}")
            ax.legend(loc="upper right")
            ax.set_title(f"{v} (profondeur ∈ [{DEPTH_TARGET-DEPTH_TOL}, {DEPTH_TARGET+DEPTH_TOL}] m) — forecast journalier")
            ax.set_xlabel("time")
            ax.set_ylabel(v)
            rmse = metrics[v]["rmse"]
            r2 = metrics[v]["r2"]
            rmse_str = f"{rmse:.4g}" if not np.isnan(rmse) else "n/a"
            r2_str = f"{r2:.4g}" if not np.isnan(r2) else "n/a"
            metrics_text = f"R² = {r2_str}    RMSE = {rmse_str}"
            fig.text(0.1, 0.02, metrics_text, ha='left', fontsize=10)
            fig.tight_layout(rect=[0, 0.04, 1, 1])
            pdf.savefig(fig)
            plt.close(fig)

        # summary page (uniquement PREDICT_VARS)
        try:
            summary_df = pd.DataFrame.from_dict({k: {"rmse": metrics[k]["rmse"], "r2": metrics[k]["r2"]} for k in PREDICT_VARS}, orient="index")
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.set_title("Résumé des metrics par variable (données filtrées par profondeur)")
            table = ax.table(cellText=np.round(summary_df.fillna(np.nan), 6).astype(object).values,
                             colLabels=summary_df.columns,
                             rowLabels=summary_df.index,
                             loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
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
                "PREDICT_VARS": PREDICT_VARS
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
            ax.text(x0, y, "paramètres :", transform=fig.transFigure, fontsize=11, weight='bold', va='top')
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
