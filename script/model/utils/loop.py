import subprocess


def run_program_in_loop(program_path, n=10):
    """
    Exécute un programme externe en boucle n fois.

    :param program_path: chemin du programme à exécuter (ex: './script.sh' ou 'python autre_script.py')
    :param n: nombre de répétitions (par défaut 10)
    """
    for i in range(1, n + 1):
        print(f"--- Exécution {i}/{n} du programme ---")
        # Lance le programme et attend qu'il se termine
        result = subprocess.run(program_path, shell=True)

        # Vérifie le code de sortie
        if result.returncode != 0:
            print(f"Le programme s'est terminé avec une erreur (code {result.returncode}) à l'exécution {i}.")
        else:
            print(f"Exécution {i} terminée avec succès.\n")


if __name__ == "__main__":
    # Exemple : exécuter "python autre_script.py" 10 fois
    run_program_in_loop("python test_LSTM_2.py", n=50)
