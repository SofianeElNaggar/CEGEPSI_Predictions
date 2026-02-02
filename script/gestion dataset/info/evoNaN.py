import re
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Charger le fichier texte
# -------------------------------------------------------------------
with open("../../results/histogrammes NaN/NaN par palier.txt", "r", encoding="utf-8") as f:
    txt = f.read()

# -------------------------------------------------------------------
# Découper le texte en blocs (paliers)
# -------------------------------------------------------------------
bloc_pattern = r"(\d+\s*-\s*\d+)\s+Résumé NaN par colonne\s*:\s*(.*?)(?=\n\d+\s*-\s*\d+|\Z)"
blocs = re.findall(bloc_pattern, txt, flags=re.S)

rows = []

for palier, contenu in blocs:
    # Extraire chaque ligne type :
    # variable   NaN_count   NaN_percent
    line_pattern = r"^([A-Za-z0-9_ ]+?)\s+(\d+)\s+([\d\.]+)$"

    for line in contenu.splitlines():
        m = re.match(line_pattern, line.strip())
        if m:
            variable = m.group(1).strip()
            nan_percent = float(m.group(3))

            rows.append([palier, variable, nan_percent])

# -------------------------------------------------------------------
# Mettre dans DataFrame
# -------------------------------------------------------------------
data = pd.DataFrame(rows, columns=["palier", "variable", "NaN_percent"])

# Trier les paliers
data["palier_start"] = data["palier"].str.extract(r"(\d+)").astype(int)
data = data.sort_values(["variable", "palier_start"])

# -------------------------------------------------------------------
# Générer un histogramme pour chaque variable
# -------------------------------------------------------------------
variables = data["variable"].unique()

for var in variables:
    subset = data[data["variable"] == var]

    plt.figure(figsize=(8, 4))
    plt.bar(subset["palier"], subset["NaN_percent"])
    plt.title(f"Évolution du % de NaN pour : {var}")
    plt.xlabel("Palier")
    plt.ylabel("% NaN")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"hist_{var.replace(' ', '_')}.png", dpi=150)
    plt.close()

print("Histogrammes générés avec succès.")

