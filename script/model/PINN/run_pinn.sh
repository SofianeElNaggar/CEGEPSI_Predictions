#!/usr/bin/env bash
# =============================================================================
# run_pinn.sh — Lance main.py dans le venv du projet PINN
#
# Usage :
#   bash run_pinn.sh [RNN_TYPE] [USE_CNN] [DECOMPOSITION_METHOD]
#
# Paramètres positionnels (tous optionnels, valeurs par défaut ci-dessous) :
#   $1  RNN_TYPE            : "GRU" | "LSTM"          (défaut : GRU)
#   $2  USE_CNN             : "true" | "false"         (défaut : true)
#   $3  DECOMPOSITION_METHOD: "VMD" | "CEEMDAN" | "SSA" | "false"  (défaut : VMD)
#
# Exemples :
#   bash run_pinn.sh
#   bash run_pinn.sh GRU false VMD
#   bash run_pinn.sh LSTM true CEEMDAN
# =============================================================================

set -euo pipefail

# ── Répertoires ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # racine de CEGEPSI_Predictions
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

# ── Paramètres (avec valeurs par défaut) ─────────────────────────────────────
RNN_TYPE="${1:-GRU}"
USE_CNN="${2:-true}"
DECOMPOSITION_METHOD="${3:-VMD}"

# ── Validation des paramètres ─────────────────────────────────────────────────
if [[ "$RNN_TYPE" != "GRU" && "$RNN_TYPE" != "LSTM" ]]; then
    echo "[ERREUR] RNN_TYPE invalide : '$RNN_TYPE'. Valeurs acceptées : GRU | LSTM"
    exit 1
fi

if [[ "$USE_CNN" != "true" && "$USE_CNN" != "false" ]]; then
    echo "[ERREUR] USE_CNN invalide : '$USE_CNN'. Valeurs acceptées : true | false"
    exit 1
fi

if [[ "$DECOMPOSITION_METHOD" != "VMD" && \
      "$DECOMPOSITION_METHOD" != "CEEMDAN" && \
      "$DECOMPOSITION_METHOD" != "SSA" && \
      "$DECOMPOSITION_METHOD" != "false" ]]; then
    echo "[ERREUR] DECOMPOSITION_METHOD invalide : '$DECOMPOSITION_METHOD'. Valeurs acceptées : VMD | CEEMDAN | SSA | false"
    exit 1
fi

# ── Vérification de l'existence du venv ──────────────────────────────────────
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "[ERREUR] Python du venv introuvable : $VENV_PYTHON"
    echo "         Assurez-vous que le venv est bien créé à la racine du projet."
    exit 1
fi

# ── Affichage du résumé ───────────────────────────────────────────────────────
echo "============================================================"
echo " Lancement du PINN"
echo "------------------------------------------------------------"
echo "  RNN_TYPE            = $RNN_TYPE"
echo "  USE_CNN             = $USE_CNN"
echo "  DECOMPOSITION_METHOD= $DECOMPOSITION_METHOD"
echo "  Python              = $VENV_PYTHON"
echo "  Répertoire          = $SCRIPT_DIR"
echo "============================================================"

# ── Lancement ─────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

export PINN_RNN_TYPE="$RNN_TYPE"
export PINN_USE_CNN="$USE_CNN"
export PINN_DECOMPOSITION_METHOD="$DECOMPOSITION_METHOD"

"$VENV_PYTHON" main.py
