# fichier: enkf_timeseries_fixed_with_pdf_depth_filtered.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

# --- nouveau: sélection autour d'une profondeur cible ---
DEPTH_COL = "depth"      # nom de la colonne profondeur dans le dataframe
DEPTH_TARGET = 15       # profondeur cible en mètres (ex: 1.0)
DEPTH_TOL = 0.2         # tolérance en mètres (ex: +/- 1.0m)

# ---------- CONFIG ----------
PARQUET_PATH = "../../dataset/OMDS-CTD data light.parquet"
TIME_COL = "time"
TRAIN_START = "2000-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2025-12-30"
ENS_SIZE = 100
Ridge_alpha = 1.0
SEED = 42
OUTPUT_PDF = "../results/prediction/enKF/v1/predictions_report_" + str(DEPTH_TARGET) + "m.pdf"
# ----------------------------
np.random.seed(SEED)
# ----------------------------

def read_and_clean(path):
    import time
    t0 = time.time()
    print("[INFO] Lecture du fichier parquet :", path)
    df = pd.read_parquet(path)
    print(f"[INFO] Lecture terminée en {time.time() - t0:.2f}s, {len(df):,} lignes, {len(df.columns)} colonnes")

    # renommer UTC -> time si nécessaire
    if "UTC" in df.columns and TIME_COL not in df.columns:
        df = df.rename(columns={"UTC": TIME_COL})

    if TIME_COL not in df.columns:
        raise KeyError(f"Colonne temporelle '{TIME_COL}' introuvable dans le fichier parquet.")

    # conversion en datetime si besoin
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        print("[INFO] Conversion de la colonne time en datetime (format initial =", df[TIME_COL].dtype, ")")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
        n_bad = df[TIME_COL].isna().sum()
        if n_bad > 0:
            print(f"[INFO] Suppression de {n_bad} lignes avec time non parseable.")
            df = df.dropna(subset=[TIME_COL])

    # mettre en index temporel trié
    df = df.set_index(TIME_COL).sort_index()
    print(f"[INFO] Nettoyage terminé en {time.time() - t0:.2f}s total")
    return df


def apply_qc_rule(df):
    print("[INFO] Application des règles QC")
    qc_cols = [c for c in df.columns if c.endswith("_QC")]
    var_cols = [c for c in df.columns if c not in qc_cols]
    print(f"[INFO] Colonnes QC détectées: {qc_cols}")
    for qc in qc_cols:
        var = qc[:-3]  # retire _QC
        if var in df.columns:
            mask_bad = df[qc].notna() & (df[qc] != 1)
            n_bad = mask_bad.sum()
            if n_bad > 0:
                print(f"[INFO] QC: {n_bad} valeurs marquées comme invalides pour '{var}' (colonne {qc}).")
                df.loc[mask_bad, var] = np.nan
    return df, var_cols

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
    # robustifier covariance: si peu d'échantillons, np.cov peut échouer -> fallback diag
    if resid.shape[0] < 2:
        Q = np.diag(np.var(resid, axis=0))
        print("[WARN] Peu d'échantillons pour estimer Q; on utilise diag(var(resid)).")
    else:
        Q = np.cov(resid.T)
    print("[INFO] A shape:", A.shape, " Q shape:", Q.shape)
    return A, Q, resid

def select_one_per_day_closest_depth(df, depth_col="depth", target=1.0, tol=1.0):
    """
    Pour chaque jour calendar, conserve l'index de la ligne dont depth_col est la plus proche de 'target'
    mais seulement si la distance absolue est <= tol. Sinon la journée est ignorée.
    Retourne DataFrame indexé par time contenant seulement ces lignes.
    Implémentation robuste: on calcule min_val = absdiff.min() puis on compare.
    """
    if depth_col not in df.columns:
        print(f"[WARN] Colonne de profondeur '{depth_col}' introuvable — on ne sous-échantillonne pas.")
        return df

    if df.shape[0] == 0:
        return df.iloc[0:0]

    tmp = df.copy()
    # enlever tz pour la date si présent
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
        # skip if min_val is NaN (defensive)
        if pd.isna(min_val):
            continue
        if min_val <= tol:
            # prendre le premier index correspondant au minimum (gère ties)
            idx_candidates = absdiff[absdiff == min_val].index
            idxmin = idx_candidates[0]
            picks.append(idxmin)
        else:
            # aucune mesure assez proche ce jour -> on ignore la journée
            continue
    if len(picks) == 0:
        print("[WARN] Aucun jour n'a de mesure dans la tolérance demandée.")
        # retourner DataFrame vide mais avec les mêmes colonnes (sans __date)
        return tmp.iloc[0:0].drop(columns="__date")
    result = tmp.loc[picks].drop(columns="__date")
    result = result.sort_index()
    print(f"[INFO] Sous-échantillonnage (tol={tol}m): {len(result)} jours retenus / {len(df)} lignes originales.")
    return result

# --- EnKF implementation ---
def enkf_forecast_and_assimilate(A, Q, R_diag, x0_mean, x0_cov, observations_df, var_cols,
                                 ens_size=100, verbose=True):
    n_vars = len(var_cols)
    print(f"[INFO] Lancement EnKF: n_vars={n_vars}, ens_size={ens_size}")
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
        print(f"[WARN] x0_cov shape attendu {(n_vars,n_vars)}, reçu {x0_cov.shape}. Recalcul avec cov(Xs) fallback.")
        x0_cov = np.eye(n_vars) * 1e-3
    x0_cov = (x0_cov + x0_cov.T) / 2.0
    x0_cov = x0_cov + 1e-6 * np.eye(n_vars)

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
    ens = ens.T  # shape (n_vars, ens)
    print("[INFO] Ensemble initialisé. Moyenne initiale (extrait):", np.round(ens.mean(axis=1)[:5], 4))

    times = observations_df.index
    analyses = []
    forecasts = []
    for t_idx in range(len(times)):
        t = times[t_idx]
        process_noise = np.random.multivariate_normal(np.zeros(n_vars), Q + 1e-8*np.eye(n_vars), size=ens_size).T
        ens = A.dot(ens) + process_noise
        forecasts.append(ens.mean(axis=1).copy())
        y_obs = observations_df.iloc[t_idx].values  # may contain NaN
        obs_mask = ~np.isnan(y_obs)
        n_obs = obs_mask.sum()
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
            print("[WARN] (Pyy + R) singulière -> ajout jitter diagonal")
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
    df_all, var_cols = apply_qc_rule(df_all)

    # Conversion explicite de la colonne depth en numérique (pour éviter comparisons entre str et float)
    if DEPTH_COL in df_all.columns:
        df_all[DEPTH_COL] = pd.to_numeric(df_all[DEPTH_COL], errors='coerce')
        n_non_numeric = df_all[DEPTH_COL].isna().sum()
        print(f"[INFO] Conversion '{DEPTH_COL}' en numérique effectuée. {n_non_numeric} valeurs invalides converties en NaN.")
    else:
        print(f"[WARN] Colonne profondeur '{DEPTH_COL}' introuvable dans le DataFrame.")

    # Retirer depth de var_cols si présent (on ne l'utilise pas comme variable d'état)
    if DEPTH_COL in var_cols:
        print(f"[INFO] Retrait de la colonne '{DEPTH_COL}' des variables d'état (utilisée seulement pour filtrage).")
        var_cols = [c for c in var_cols if c != DEPTH_COL]

    # convertir les colonnes d'état en numériques
    if len(var_cols) == 0:
        raise ValueError("Aucune variable d'état détectée après retrait des colonnes QC/depth.")
    df_all[var_cols] = df_all[var_cols].apply(pd.to_numeric, errors="coerce")
    print(f"[INFO] Variables détectées ({len(var_cols)}): {var_cols[:10]}{'...' if len(var_cols)>10 else ''}")

    # Appliquer le filtrage de profondeur (en s'assurant que la colonne depth est numérique)
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
        raise ValueError(
            f"Aucune ligne retenue après filtrage par profondeur (DEPTH_TARGET={DEPTH_TARGET}, DEPTH_TOL={DEPTH_TOL}). "
            "Ajuste DEPTH_TOL, ou vérifie la colonne 'depth' dans le fichier parquet."
        )

    # split train/test from the FILTERED dataframe (on times and variables)
    train = df_filtered.loc[TRAIN_START:TRAIN_END, var_cols].copy()
    test  = df_filtered.loc[TEST_START:TEST_END, var_cols].copy()
    print(f"[INFO] Train (après filtrage): {train.shape[0]} lignes, Test (après filtrage): {test.shape[0]} lignes")

    # Forcer la sélection 1 mesure/jour dans la période test si possible
    test_full_period = df_filtered.loc[TEST_START:TEST_END]  # on cherche dans le sous-ensemble filtré
    if test_full_period.shape[0] == 0:
        raise ValueError("Aucune donnée disponible dans la période TEST après filtrage profondeur.")
    # select_one_per_day_closest_depth attend la colonne depth; on lui passe test_full_period (toutes colonnes)
    test_subsampled = select_one_per_day_closest_depth(test_full_period, depth_col=DEPTH_COL, target=DEPTH_TARGET, tol=DEPTH_TOL)
    # si la fonction retourne un DataFrame vide, on garde test (éventuellement vide) pour faire échouer plus tard proprement
    if test_subsampled.shape[0] == 0:
        print("[WARN] La sous-sélection journalière n'a retourné aucune ligne (tol peut-être trop stricte). On conserve l'ensemble test filtré sans sous-échantillonnage.")
        test = test.copy()
    else:
        test = test_subsampled[var_cols].copy()
    print(f"[INFO] Après sous-échantillonnage journalier (et filtrage depth): Test: {test.shape[0]} lignes (une par jour si possible).")

    # Vérifications avant entraînement
    if train.dropna(how='all').shape[0] < 2:
        raise ValueError(
            f"Pas assez de données d'entraînement après filtrage profondeur (lignes retenues: {train.shape[0]}). "
            "Augmente DEPTH_TOL, élargis TRAIN_START/TRAIN_END, ou vérifie la colonne 'depth'."
        )

    # build training pairs
    Xs, Xn = build_training_pairs(train)
    A, Q, resid = estimate_A(Xs, Xn, alpha=Ridge_alpha)
    print("Estimated A shape:", A.shape)
    print("Estimated process covariance Q shape:", Q.shape)

    # Observation noise R diag: estimate variance of residuals per variable of train (or set small)
    R_diag = np.var(resid, axis=0)
    print("[INFO] R_diag (extrait):", np.round(R_diag[:5], 6))

    # initial state for enkf: take first valid vector in test (ou fallback vers la moyenne du train)
    if test.dropna(how='all').shape[0] == 0:
        # fallback: essayer de prendre la dernière ligne du train comme état initial
        print("[WARN] Pas d'observations valides dans la période test (après filtrage profondeur). Utilisation de la dernière ligne du train pour x0.")
        last_train_valid = train.dropna(how='all')
        if last_train_valid.shape[0] == 0:
            raise ValueError("Ni test ni train ne contiennent d'observations valides après filtrage.")
        first_valid_idx = last_train_valid.index[-1]
        x0_mean = last_train_valid.loc[first_valid_idx].fillna(train.mean()).values
    else:
        first_valid_idx = test.dropna(how='all').index[0]
        x0_mean = test.loc[first_valid_idx].fillna(train.mean()).values

    x0_mean = np.asarray(x0_mean).ravel()
    if x0_mean.shape[0] != len(var_cols):
        print("[WARN] Taille de x0_mean incohérente, on recalcule à partir de la moyenne du train")
        x0_mean = train.mean().values

    x0_cov = np.cov(Xs.T) * 0.5 + 1e-6*np.eye(len(var_cols))
    print("[INFO] x0_mean extrait, x0_cov shape:", x0_cov.shape)

    # run enkf
    test_obs = test.copy()
    print("[INFO] Lancement de l'EnKF sur la période test (données filtrées par profondeur)...")
    df_f, df_a = enkf_forecast_and_assimilate(A, Q, R_diag, x0_mean, x0_cov, test_obs, var_cols, ens_size=ENS_SIZE)

    # compute RMSE and R2 only where observations exist
    metrics = {}
    for v in var_cols:
        if v not in test.columns or v not in df_a.columns:
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

    print("[INFO] Metrics calculés pour chaque variable (extrait):")
    for k in list(metrics.keys())[:10]:
        print(f" {k}: RMSE={metrics[k]['rmse']:.4g}, R2={metrics[k]['r2']:.4g}")

    # Save plots to PDF: one page per variable
    print(f"[INFO] Sauvegarde des graphiques et metrics dans le PDF: {OUTPUT_PDF}")
    with PdfPages(OUTPUT_PDF) as pdf:
        for v in var_cols:
            fig, ax = plt.subplots(figsize=(10, 4))
            if (~test[v].isna()).any():
                ax.plot(test.index, test[v], marker='.', linestyle='none', label=f"obs {v}")
            if v in df_f.columns:
                ax.plot(df_f.index, df_f[v], label=f"forecast {v}")
            if v in df_a.columns:
                ax.plot(df_a.index, df_a[v], label=f"analysis {v}")
            ax.legend(loc="upper right")
            ax.set_title(f"{v} (données profondeur ∈ [{DEPTH_TARGET-DEPTH_TOL}, {DEPTH_TARGET+DEPTH_TOL}] m)")
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

    print(f"[INFO] PDF sauvegardé: {OUTPUT_PDF}")
    print("[END] exécution terminée")
    return {"A": A, "Q": Q, "forecast": df_f, "analysis": df_a, "metrics": metrics}

if __name__ == "__main__":
    out = main()
