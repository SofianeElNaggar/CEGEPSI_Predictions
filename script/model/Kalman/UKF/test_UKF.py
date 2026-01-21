# fichier: ukf_timeseries_with_depth_filter.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

# ---------- CONFIG ----------
PARQUET_PATH = "../../dataset/OMDS-CTD data light.parquet"
TIME_COL = "time"
TRAIN_START = "2000-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2025-12-30"
ENS_SIZE = 100            # (non utilisé par UKF, conservé pour compatibilité)
Ridge_alpha = 1.0
SEED = 42
OUTPUT_PDF = "predictions_report_ukf.pdf"
# --- profondeur ---
DEPTH_COL = "depth"
DEPTH_TARGET = 1.0
DEPTH_TOL = 0.1
# --- UKF params (Van der Merwe) ---
UKF_ALPHA = 1e-3
UKF_BETA  = 2.0
UKF_KAPPA = 0.0
# ----------------------------
np.random.seed(SEED)
# ----------------------------

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
            mask_bad = df[qc].notna() & (df[qc] != 1)
            n_bad = mask_bad.sum()
            if n_bad > 0:
                print(f"[INFO] QC: {n_bad} valeurs invalides pour '{var}' (col {qc}).")
                df.loc[mask_bad, var] = np.nan
    return df, var_cols

def build_training_pairs(X):
    print("[INFO] Construction des paires d'entraînement (x_t, x_{t+1})")
    Xs, Xn = [], []
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

def estimate_A(Xs, Xn, alpha=1.0):
    print("[INFO] Estimation de A par Ridge (fit_intercept=False)")
    n_vars = Xs.shape[1]
    A = np.zeros((n_vars, n_vars))
    model = Ridge(alpha=alpha, fit_intercept=False)
    for j in range(n_vars):
        y = Xn[:, j]
        model.fit(Xs, y)
        A[j, :] = model.coef_
    resid = Xn - Xs.dot(A.T)
    if resid.shape[0] < 2:
        Q = np.diag(np.var(resid, axis=0))
        print("[WARN] Peu d'échantillons pour estimer Q; fallback diag(var).")
    else:
        Q = np.cov(resid.T)
    print("[INFO] A shape:", A.shape, "Q shape:", Q.shape)
    return A, Q, resid

# ------------------- UKF helpers -------------------
def compute_sigma_points(x, P, alpha=1e-3, beta=2.0, kappa=0.0):
    """
    Retourne (sigma_points, Wm, Wc)
    sigma_points: shape (2n+1, n)
    Wm: weights mean (2n+1,)
    Wc: weights cov (2n+1,)
    """
    x = np.asarray(x).ravel()
    P = np.asarray(P)
    n = x.size
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    # try cholesky, fallback to eigh
    try:
        A = np.linalg.cholesky(c * P)
    except np.linalg.LinAlgError:
        # make P symmetric and add tiny jitter
        P = (P + P.T) / 2.0 + 1e-8 * np.eye(n)
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals[eigvals < 0] = 0.0
        A = eigvecs @ np.diag(np.sqrt(eigvals * c))
    sigma_points = np.zeros((2*n + 1, n))
    sigma_points[0] = x
    for i in range(n):
        col = A[:, i]
        sigma_points[1 + i]     = x + col
        sigma_points[1 + n + i] = x - col
    Wm = np.full(2*n + 1, 1.0 / (2.0 * c))
    Wc = np.full(2*n + 1, 1.0 / (2.0 * c))
    Wm[0] = lam / c
    Wc[0] = lam / c + (1 - alpha**2 + beta)
    return sigma_points, Wm, Wc

def ukf_predict(A_mat, x_mean, x_cov, Q, alpha=UKF_ALPHA, beta=UKF_BETA, kappa=UKF_KAPPA):
    """
    Predict step of UKF for linear dynamic x_{k+1} = A x_k + noise (noise additive).
    Returns x_mean_pred, P_pred, sigma_points_pred (for possible use).
    """
    n = x_mean.size
    sigma_pts, Wm, Wc = compute_sigma_points(x_mean, x_cov, alpha, beta, kappa)
    # propagate sigma points through linear map x' = A x
    sigma_pts_prop = (sigma_pts @ A_mat.T)  # shape (2n+1, n) because A_mat @ x (n,n) dot (n,) -> (n,)
    # compute predicted mean
    x_pred = np.sum(Wm[:, None] * sigma_pts_prop, axis=0)
    # predicted covariance
    diff = sigma_pts_prop - x_pred[None, :]
    P_pred = np.zeros((n, n))
    for i in range(2*n + 1):
        P_pred += Wc[i] * np.outer(diff[i], diff[i])
    P_pred = (P_pred + P_pred.T) / 2.0
    P_pred = P_pred + Q  # add process noise (additive)
    return x_pred, P_pred, sigma_pts_prop, Wm, Wc

def ukf_update(x_pred, P_pred, sigma_pts_prop, Wm, Wc, y_obs, R_diag, obs_idx):
    """
    UKF update for observations y = H x + noise, but we do it by selecting components obs_idx.
    - x_pred: predicted state mean (n,)
    - P_pred: predicted covariance (n,n)
    - sigma_pts_prop: sigma points after propagation (2n+1, n)
    - Wm, Wc: weights
    - y_obs: observed vector (length n_obs) with actual values
    - R_diag: full R diag (len n) -> we will use entries at obs_idx to form R_obs
    - obs_idx: indices of observed state variables (array-like)
    Returns x_upd, P_upd
    """
    n = x_pred.size
    obs_idx = np.asarray(obs_idx, dtype=int)
    n_obs = obs_idx.size
    # map sigma points to observation space: take components obs_idx
    Y = sigma_pts_prop[:, obs_idx]  # shape (2n+1, n_obs)
    # predicted obs mean
    y_mean = np.sum(Wm[:, None] * Y, axis=0)
    # obs covariance Pyy and cross-covariance Pxy
    Pyy = np.zeros((n_obs, n_obs))
    Pxy = np.zeros((n, n_obs))
    diffY = Y - y_mean[None, :]
    diffX = sigma_pts_prop - x_pred[None, :]
    for i in range(Y.shape[0]):
        Pyy += Wc[i] * np.outer(diffY[i], diffY[i])
        Pxy += Wc[i] * np.outer(diffX[i], diffY[i])
    # add observation noise
    R_obs = np.diag(np.asarray(R_diag)[obs_idx])
    Pyy = (Pyy + Pyy.T) / 2.0 + R_obs
    # Kalman gain
    # regularize Pyy if needed
    try:
        K = Pxy @ np.linalg.inv(Pyy)
    except np.linalg.LinAlgError:
        K = Pxy @ np.linalg.inv(Pyy + 1e-8 * np.eye(n_obs))
    # innovation
    innov = y_obs - y_mean
    x_upd = x_pred + K @ innov
    P_upd = P_pred - K @ Pyy @ K.T
    P_upd = (P_upd + P_upd.T) / 2.0
    # ensure positive definite-ish
    jitter = 1e-12 * np.eye(n)
    P_upd = P_upd + jitter
    return x_upd, P_upd

# ------------------- UKF main routine -------------------
def ukf_forecast_and_assimilate(A_mat, Q, R_diag, x0_mean, x0_cov, observations_df, var_cols,
                                alpha=UKF_ALPHA, beta=UKF_BETA, kappa=UKF_KAPPA, verbose=True):
    """
    A_mat: matrix (n x n) estimated from training
    Q: process covariance (n x n)
    R_diag: vector length n (observation noise variance per state variable)
    x0_mean: initial mean (n,)
    x0_cov: initial covariance (n x n)
    observations_df: DataFrame indexed by time with columns = var_cols (observations, may contain NaN)
    var_cols: list of variable names in state (order must match A_mat)
    Returns df_forecast, df_analysis (DataFrames indexed by times)
    """
    n = len(var_cols)
    x_mean = np.asarray(x0_mean).ravel()
    P = np.asarray(x0_cov)
    # defensive shapes
    if x_mean.size != n:
        raise ValueError("x0_mean length incompatible with var_cols")
    if P.shape != (n, n):
        P = np.eye(n) * 1e-6

    times = observations_df.index
    forecasts = []
    analyses = []

    for t_idx, t in enumerate(times):
        # predict
        x_pred, P_pred, sigma_pts_prop, Wm, Wc = ukf_predict(A_mat, x_mean, P, Q, alpha, beta, kappa)
        forecasts.append(x_pred.copy())

        # observation at t
        y_obs_full = observations_df.iloc[t_idx].values  # length n, contains NaN for missing
        obs_mask = ~np.isnan(y_obs_full)
        n_obs = int(obs_mask.sum())
        if verbose:
            print(f"[UKF STEP {t_idx+1}/{len(times)}] time={t}, obs_count={n_obs}")
        if n_obs == 0:
            # no update
            x_mean = x_pred
            P = P_pred
            analyses.append(x_mean.copy())
            continue

        obs_idx = np.where(obs_mask)[0]
        y_obs = y_obs_full[obs_idx]
        # update only on observed components
        x_mean, P = ukf_update(x_pred, P_pred, sigma_pts_prop, Wm, Wc, y_obs, R_diag, obs_idx)
        analyses.append(x_mean.copy())

    df_forecast = pd.DataFrame(np.vstack(forecasts), index=times, columns=var_cols)
    df_analysis = pd.DataFrame(np.vstack(analyses), index=times, columns=var_cols)
    return df_forecast, df_analysis

# ------------------- depth selection helper (unchanged robust) -------------------
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

# ------------------- main -------------------
def main():
    print("[START] UKF pipeline")
    df_all = read_and_clean(PARQUET_PATH)
    df_all, var_cols = apply_qc_rule(df_all)

    # convert depth to numeric
    if DEPTH_COL in df_all.columns:
        df_all[DEPTH_COL] = pd.to_numeric(df_all[DEPTH_COL], errors='coerce')
        print(f"[INFO] Conversion '{DEPTH_COL}' -> numérique (NaN si invalide).")
    else:
        print(f"[WARN] Colonne '{DEPTH_COL}' absente.")

    # remove depth from state variables (we use it only to filter)
    if DEPTH_COL in var_cols:
        var_cols = [c for c in var_cols if c != DEPTH_COL]
        print(f"[INFO] '{DEPTH_COL}' retirée de var_cols (état).")

    # ensure numeric state columns
    if len(var_cols) == 0:
        raise ValueError("Aucune variable d'état après retrait de depth/QC.")
    df_all[var_cols] = df_all[var_cols].apply(pd.to_numeric, errors='coerce')
    print(f"[INFO] Variables d'état: {var_cols[:10]}{'...' if len(var_cols)>10 else ''}")

    # filter by depth interval
    if DEPTH_COL in df_all.columns:
        lower = DEPTH_TARGET - DEPTH_TOL
        upper = DEPTH_TARGET + DEPTH_TOL
        depth_mask = df_all[DEPTH_COL].notna() & (df_all[DEPTH_COL] >= lower) & (df_all[DEPTH_COL] <= upper)
        df_filtered = df_all.loc[depth_mask].copy()
        print(f"[INFO] Filtrage depth ±{DEPTH_TOL}m -> {len(df_filtered)} lignes retenues.")
    else:
        df_filtered = df_all.copy()

    if df_filtered.shape[0] == 0:
        raise ValueError("Aucune ligne après filtrage depth. Ajuste DEPTH_TOL ou vérifie la colonne depth.")

    # split train/test on filtered df
    train = df_filtered.loc[TRAIN_START:TRAIN_END, var_cols].copy()
    test_full = df_filtered.loc[TEST_START:TEST_END]  # keep all cols to use depth for subsample
    print(f"[INFO] Train (filtré): {train.shape[0]} lignes, Test full (filtré): {test_full.shape[0]} lignes")

    # subsample test to 1 per day close to depth target
    test_sub = select_one_per_day_closest_depth(test_full, depth_col=DEPTH_COL, target=DEPTH_TARGET, tol=DEPTH_TOL)
    if test_sub.shape[0] == 0:
        print("[WARN] Sous-échantillonnage produit 0 lignes -> on utilisera l'ensemble test filtré (sans daily subsample).")
        test = test_full[var_cols].copy()
    else:
        test = test_sub[var_cols].copy()
    print(f"[INFO] Test final utilisé pour assimilation: {test.shape[0]} lignes")

    # training pairs and estimate A, Q
    if train.dropna(how='all').shape[0] < 2:
        raise ValueError("Pas assez de données d'entraînement après filtrage depth.")
    Xs, Xn = build_training_pairs(train)
    A, Q, resid = estimate_A(Xs, Xn, alpha=Ridge_alpha)

    # R diag (observation noise) estimated from residuals
    R_diag = np.var(resid, axis=0)
    # ensure R_diag length matches
    if R_diag.size != len(var_cols):
        # fallback small diag
        R_diag = np.ones(len(var_cols)) * 1e-3
    print("[INFO] R_diag sample:", np.round(R_diag[:5], 6))

    # initial state: use first valid in test or last train
    if test.dropna(how='all').shape[0] > 0:
        idx0 = test.dropna(how='all').index[0]
        x0_mean = test.loc[idx0].fillna(train.mean()).values
    else:
        print("[WARN] Aucun point valide dans test; on prend la dernière ligne du train pour init.")
        last_valid = train.dropna(how='all')
        if last_valid.shape[0] == 0:
            raise ValueError("Ni test ni train ne contiennent d'observations valides.")
        x0_mean = last_valid.iloc[-1].fillna(train.mean()).values

    x0_mean = np.asarray(x0_mean).ravel()
    if x0_mean.size != len(var_cols):
        x0_mean = train.mean().values

    # initial covariance: use cov(Xs) fallback
    try:
        x0_cov = np.cov(Xs.T) * 0.5 + 1e-6 * np.eye(len(var_cols))
    except Exception:
        x0_cov = np.eye(len(var_cols)) * 1e-3
    print("[INFO] x0_mean shape:", x0_mean.shape, " x0_cov shape:", x0_cov.shape)

    # run UKF
    print("[INFO] Lancement UKF...")
    df_f, df_a = ukf_forecast_and_assimilate(A, Q, R_diag, x0_mean, x0_cov, test, var_cols,
                                             alpha=UKF_ALPHA, beta=UKF_BETA, kappa=UKF_KAPPA, verbose=True)

    # metrics
    metrics = {}
    for v in var_cols:
        if v not in test.columns:
            metrics[v] = {"rmse": np.nan, "r2": np.nan}
            continue
        mask = ~test[v].isna()
        if mask.sum() == 0:
            metrics[v] = {"rmse": np.nan, "r2": np.nan}
            continue
        y_true = test.loc[mask, v].values
        y_pred = df_a.loc[mask, v].values
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = np.nan
        metrics[v] = {"rmse": rmse, "r2": r2}

    # Save PDF plots
    print(f"[INFO] Sauvegarde PDF: {OUTPUT_PDF}")
    with PdfPages(OUTPUT_PDF) as pdf:
        for v in var_cols:
            fig, ax = plt.subplots(figsize=(10, 4))
            if v in test.columns and (~test[v].isna()).any():
                ax.plot(test.index, test[v], marker='.', linestyle='none', label=f"obs {v}")
            if v in df_f.columns:
                ax.plot(df_f.index, df_f[v], label=f"forecast {v}")
            if v in df_a.columns:
                ax.plot(df_a.index, df_a[v], label=f"analysis {v}")
            ax.legend(loc="upper right")
            ax.set_title(f"{v} (depth ∈ [{DEPTH_TARGET-DEPTH_TOL}, {DEPTH_TARGET+DEPTH_TOL}] m) — UKF")
            ax.set_xlabel("time")
            ax.set_ylabel(v)
            rmse = metrics[v]["rmse"]
            r2 = metrics[v]["r2"]
            rmse_str = f"{rmse:.4g}" if not np.isnan(rmse) else "n/a"
            r2_str = f"{r2:.4g}" if not np.isnan(r2) else "n/a"
            fig.text(0.1, 0.02, f"R² = {r2_str}    RMSE = {rmse_str}", ha='left', fontsize=10)
            fig.tight_layout(rect=[0, 0.04, 1, 1])
            pdf.savefig(fig)
            plt.close(fig)
        # summary
        try:
            summary_df = pd.DataFrame.from_dict({k: {"rmse": metrics[k]["rmse"], "r2": metrics[k]["r2"]} for k in var_cols}, orient="index")
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.set_title("Résumé des metrics par variable (UKF)")
            table = ax.table(cellText=np.round(summary_df.fillna(np.nan), 6).astype(object).values,
                             colLabels=summary_df.columns, rowLabels=summary_df.index, loc='center', cellLoc='center')
            table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.2)
            pdf.savefig(fig); plt.close(fig)
        except Exception as e:
            print("[WARN] Impossible d'ajouter la page de résumé:", e)

    print("[END] UKF pipeline — PDF saved:", OUTPUT_PDF)
    return {"A": A, "Q": Q, "forecast": df_f, "analysis": df_a, "metrics": metrics}

if __name__ == "__main__":
    out = main()
