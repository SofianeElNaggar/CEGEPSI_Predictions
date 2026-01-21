import subprocess

scripts = ["enrich_weather_openmeteo.py", "enrich_precip_openmeteo.py", "enrich_wind_openmeteo.py"]

for script in scripts:
    print(f"➡️ Lancement de {script} ...")

    # Lance le script et attend qu'il se termine
    result = subprocess.run(["python", script])

    # Vérifie si le script s'est correctement exécuté
    if result.returncode != 0:
        print(f"❌ Erreur dans {script}, arrêt de la chaîne.")
        break

print("✔️ Tous les scripts terminés.")
