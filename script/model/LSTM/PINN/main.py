# main.py
# Point d'entrée: exécution unique sur ALL_TARGETS
from pipeline import run_pipeline
from utils import ALL_TARGETS, OUTPUT_PDF_TEMPLATE, INPUT_ONLY_COLS, TIME_FEATURE_COLS

def main():
    target_cols = ALL_TARGETS.copy()
    print(f"Exécution unique sur la combinaison (ALL_TARGETS) : {target_cols}")
    print(f"Colonnes présentes en entré (INPUT_ONLY_COLS) : {INPUT_ONLY_COLS}")
    print(f"Colonnes temporelles (TIME_FEATURE_COLS) : {TIME_FEATURE_COLS}")
    try:
        run_pipeline(target_cols, OUTPUT_PDF_TEMPLATE)
    except Exception as e:
        import traceback
        print(f"Erreur lors de l'exécution pour {target_cols}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
