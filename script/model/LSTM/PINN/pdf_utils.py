# pdf_utils.py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re
from data_utils import get_next_pdf_path

def save_results_pdf(out_template, target_cols, feature_cols, dates, true_targets, transformed_preds, best_params, rmses, r2s, train_df, test_df):
    names_safe = "_".join([re.sub(r'[^A-Za-z0-9]+', '', c) for c in target_cols])
    out_template = out_template.format(names=names_safe)
    out_pdf = get_next_pdf_path(out_template)
    try:
        with PdfPages(out_pdf) as pdf:
            fig_all, axs = plt.subplots(nrows=len(target_cols), ncols=1, figsize=(10, 3 * len(target_cols)), constrained_layout=True)
            if len(target_cols) == 1:
                axs = [axs]
            for i, col in enumerate(target_cols):
                ax = axs[i]
                ax.plot(dates, true_targets[:, i], label='Réel')
                ax.plot(dates, transformed_preds[:, i], label=f'Prédit (x={best_params[col][0]:.4f}, y={best_params[col][1]:.4f})')
                ax.set_title(f"{col} — RMSE={rmses[col]:.3f}  R2={r2s[col]:.3f}")
                ax.legend()
                ax.set_ylabel(col)
            pdf.savefig(fig_all)
            plt.close(fig_all)

            fig_sum = plt.figure(figsize=(8.27, 11.69))
            fig_sum.clf()
            txt = "Résultats LSTM multivarié (après OLS)\n\n"
            txt += (f"Valeur d'entré uniquement : "
                    f"\n{feature_cols[:5]}\n{feature_cols[5:]}\n\n")
            txt += f"Valeur d'entré et de sortie : {target_cols}\n\n"
            txt += f"Période entraînement : {train_df.index.min().date()} -> {train_df.index.max().date()}\n"
            txt += f"Période test : {test_df.index.min().date()} -> {test_df.index.max().date()}\n\n"
            txt += "Metrics par variable (après transformation optimale) :\n"
            for col in target_cols:
                bx, by, br2 = best_params[col]
                txt += f" - {col}: RMSE={rmses[col]:.4f}, R2={r2s[col]:.4f}  (x={bx:.6f}, y={by:.6f}, R2_opt={br2:.6f})\n"
            fig_sum.text(0.01, 0.99, txt, fontsize=10, va='top')
            pdf.savefig()
            plt.close(fig_sum)
        print(f"PDF saved: {out_pdf}")
    except Exception as e:
        print("Erreur lors de la sauvegarde du PDF:", e)
        raise
