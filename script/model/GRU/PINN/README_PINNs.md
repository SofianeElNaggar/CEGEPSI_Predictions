# Équations et logique des PINNs

Ce document décrit les **Physics-Informed Neural Networks** utilisées dans le pipeline LSTM : **DissolvedOxygenPINN** (O₂ dissous) et **pHPINN** (pH via le carbone inorganique dissous, DIC).

---

## 1. DissolvedOxygenPINN — Oxygène dissous

### 1.1 Idée physique

Le PINN impose que l’évolution temporelle de la concentration en O₂ dissous (DO) respecte un **bilan de masse** dans la couche de surface (bilan de type métabolisme écosystémique, *cf.* Hornbach et al.) :

- **Sources** : photosynthèse, réaération air–eau  
- **Puits** : respiration  

Toutes les variables sont lues dans les **entrées** de la séquence (pas dans les prédictions). La dérivée temporelle est approchée par une différence finie le long de la fenêtre temporelle.

### 1.2 Équation de bilan

\[
\frac{\mathrm{d}[\mathrm{DO}]}{\mathrm{d}t}
\;=\;
P \;-\; R \;+\; k_{\mathrm{rea}} \,\big( [\mathrm{DO}]_{\mathrm{sat}} - [\mathrm{DO}] \big)
\]

- \([\mathrm{DO}]\) : O₂ dissous (ml/L)  
- \(P\) : production par photosynthèse  
- \(R\) : consommation par respiration  
- \(k_{\mathrm{rea}}\) : coefficient de réaération air–eau  
- \([\mathrm{DO}]_{\mathrm{sat}}\) : concentration de saturation en O₂  

Unité de temps : **jour** (Δt = 1 jour entre deux pas).

### 1.3 Formules des termes

#### Photosynthèse

\[
P \;=\; \alpha_P \cdot \mathrm{Chl} \cdot \exp(0.07 \cdot T_w)
\]

- \(\alpha_P\) : paramètre appris  
- \(\mathrm{Chl}\) : chlorophylle (mg/m³), lue dans les entrées  
- \(T_w\) : température de l’eau (°C)  

La photosynthèse augmente avec la biomasse (Chl) et la température.

#### Respiration

\[
R \;=\; \alpha_R \cdot \exp(\beta_R \cdot T_w)
\]

- \(\alpha_R,\, \beta_R\) : paramètres appris  
- \(T_w\) : température de l’eau  

La respiration ne dépend que de la température (pas de Chl ni de DO dans cette formulation).

#### Réaération air–eau

\[
k_{\mathrm{rea}}
\;=\;
\alpha_k \,\Big( W + 0.5\,T_{\mathrm{tide}} \Big)
\,\Big( 1 + \gamma_{\mathrm{air}} \, \big| T_a - T_w \big| \Big)
\]

- \(\alpha_k,\, \gamma_{\mathrm{air}}\) : paramètres appris  
- \(W\) : vent (km/h), *Spd of Max Gust*  
- \(T_{\mathrm{tide}}\) : amplitude de marée (m)  
- \(T_a\) : température de l’air (°C), \(T_w\) : température de l’eau  

Le gradient thermique air–eau modifie l’efficacité du transfert.

#### Saturation en O₂ (Weiss 1970)

\[
\ln [\mathrm{DO}]_{\mathrm{sat}}
\;=\;
A_1 + A_2\,\frac{100}{T_k} + A_3\,\ln\frac{T_k}{100} + A_4\,\frac{T_k}{100}
\;+\;
S\,\Big( B_1 + B_2\,\frac{T_k}{100} + B_3\,\Big(\frac{T_k}{100}\Big)^2 \Big)
\]

avec \(T_k = T_w + 273.15\) (K) et \(S\) la salinité (PSS-78). Les constantes \(A_i,\, B_i\) sont fixées (Weiss 1970). \([\mathrm{DO}]_{\mathrm{sat}}\) en ml/L.

### 1.4 Discretisation et résidu

On travaille sur la séquence d’entrée \((t_0,\, \ldots,\, t_{L-1})\). Pour chaque paire de pas consécutifs :

\[
\frac{\mathrm{d}[\mathrm{DO}]}{\mathrm{d}t}\bigg|_{t \,\approx\, t_{k}}
\;\approx\;
[\mathrm{DO}]_{k+1} - [\mathrm{DO}]_k
\;=\;
\Delta[\mathrm{DO}]_k
\]

Les termes \(P,\, R,\, k_{\mathrm{rea}},\, [\mathrm{DO}]_{\mathrm{sat}}\) sont évalués au pas \(k\) (avant le saut).

Le **résidu** au pas \(k\) est :

\[
r_k \;=\; \Delta[\mathrm{DO}]_k \;-\; \Big( P - R + k_{\mathrm{rea}}\,([\mathrm{DO}]_{\mathrm{sat}} - [\mathrm{DO}]_k) \Big)
\]

### 1.5 Loss PINN

\[
\mathcal{L}_{\mathrm{DO}}
\;=\;
\lambda \cdot \frac{1}{N} \sum_{k} r_k^2
\]

avec \(\lambda\) le `weight` du PINN et \(N\) le nombre de résidus (sur tous les pas et tous les échantillons du batch).

### 1.6 Variables requises (entrées)

| Variable        | Rôle principal                         |
|-----------------|----------------------------------------|
| DO              | O₂ dissous (ml/L) — cible + entrée     |
| \(T_w\)         | Température eau (°C)                   |
| \(T_a\)         | Température air (°C)                   |
| Chl             | Chlorophylle (mg/m³)                   |
| \(W\)           | Vent (km/h)                            |
| \(T_{\mathrm{tide}}\) | Amplitude de marée (m)          |
| \(S\)           | Salinité (PSS-78)                      |

### 1.7 Paramètres appris

- \(\alpha_P,\, \alpha_R,\, \beta_R,\, \alpha_k,\, \gamma_{\mathrm{air}}\)

---

## 2. pHPINN — pH et carbone inorganique dissous (DIC)

### 2.1 Idée physique

Le pH de l’eau de mer est lié au **système des carbonates** et au **bilan de carbone inorganique dissous (DIC)**. Le PINN :

1. Reconstruit le **DIC** à partir du pH (et de l’alcalinité) via les équilibres carbonates.  
2. Impose que l’évolution du DIC entre le **dernier pas d’entrée** et le **pas prédit** suive un bilan :  
   - **Puits** : photosynthèse (consommation de CO₂/DIC)  
   - **Sources** : respiration, flux air–mer de CO₂  

Seule la **transition** (dernier pas de la séquence → pas suivant, prédit par le modèle) est contrainte, et la prédiction de pH du LSTM intervient dans le calcul du DIC au pas prédit.

### 2.2 Équation de bilan du DIC

\[
\frac{\mathrm{d}[\mathrm{DIC}]}{\mathrm{d}t}
\;=\;
-P \;+\; R \;+\; F
\]

- \([\mathrm{DIC}]\) : carbone inorganique dissous (µmol/L ou µmol/kg)  
- \(P\) : consommation de DIC par photosynthèse (µmol/L/jour)  
- \(R\) : production de DIC par respiration (µmol/L/jour)  
- \(F\) : flux air–mer de CO₂ (entrant = augmentation de DIC), en µmol/L/jour  

### 2.3 Système carbonate (équilibres)

#### Constantes K₁, K₂ (Lueker et al. 2000)

On utilise les formulations de **Lueker et al. (2000)** pour \(K_1\) et \(K_2\), valides pour l'eau de mer.

\[
\mathrm{p}K_1 \;=\; \frac{3633.86}{T_K} - 61.2172 + 9.67770 \ln T_K - 0.011555 S + 0.0001152 S^2
\]

\[
\mathrm{p}K_2 \;=\; \frac{471.78}{T_K} + 25.9290 - 3.16967 \ln T_K - 0.01781 S + 0.0001122 S^2
\]

avec \(T_K\) en Kelvin et \(S\) en PSU.

#### Alcalinité des Borates (Dickson 1990)

L'alcalinité totale (TA) inclut l'alcalinité des borates (BA).

\[ \mathrm{TA}_{carb} = \mathrm{TA} - \mathrm{BA} \]

\[ \mathrm{BA} = \frac{[B]_T \cdot K_B}{K_B + [\mathrm{H}^+]} \]

- \([B]_T\) (Bore total) calculé avec le ratio \(0.1324\) mg/kg/PSU selon **Lee et al. (2010)**.
- \(K_B\) calculé selon **Dickson (1990)**.


#### DIC à partir de TA et [H⁺] (ou pH)

À partir de TA et \([\mathrm{H}^+] = 10^{-\mathrm{pH}}\) :

\[
[\mathrm{DIC}]
\;=\;
\mathrm{TA} \cdot
\frac{ \displaystyle \frac{[\mathrm{H}^+]}{K_1} + 1 + \frac{K_2}{[\mathrm{H}^+]} }
     { \displaystyle 1 + \frac{2\,K_2}{[\mathrm{H}^+]} }
\]

C’est la fonction `dic_from_ph_ta(TA, H, K1, K2)`.

#### [CO₂] à partir de DIC et [H⁺]

\[
[\mathrm{CO}_2]
\;=\;
\frac{[\mathrm{DIC}]_{\mathrm{mol}}}
     { 1 + \dfrac{K_1}{[\mathrm{H}^+]} + \dfrac{K_1 K_2}{[\mathrm{H}^+]^2} }
\]

avec \([\mathrm{DIC}]_{\mathrm{mol}} = [\mathrm{DIC}]_{\mu\mathrm{mol}}\,/\,10^6\). Utilisée pour le flux air–mer (`co2_from_dic_h`).

### 2.4 Alcalinité totale (TA)

Formule polynomiale en \(T\) et \(S\) (référence 20 °C, 35 PSU), inspirée des relations globales TA–salinité–température (Lee et al.)) :

\[ \mathrm{TA} = 2305 + 53.97\,(S-35) + 2.74\,(S-35)^2 - 1.16\,(T-20) - 0.040\,(T-20)^2 \quad (\mu\mathrm{mol}/\mathrm{kg}) \]

Formule selon **Lee et al. 2006**

### 2.5 Solubilité du CO₂ (K₀, Weiss 1974)

\[ \ln K_0 = A_1 + A_2\,\frac{100}{T_k} + A_3\,\ln\frac{T_k}{100} + S\,\left( B_1 + B_2\,\frac{T_k}{100} + B_3\,\left(\frac{T_k}{100}\right)^2 \right) \]

\(K_0\) en mol/(L·atm). Concentrations à l’équilibre :

\[
[\mathrm{CO}_2]_{\mathrm{eq}} \;=\; K_0 \cdot p_{\mathrm{CO}_2}^{\mathrm{atm}}
\]

avec \(p_{\mathrm{CO}_2}^{\mathrm{atm}}\) en atm (le paramètre `pCO2_air` est en µatm, multiplié par \(10^{-6}\)).

### 2.6 Termes du bilan (sur le dernier pas de la séquence)

#### Photosynthèse

\[
P \;=\; \alpha_P \cdot \mathrm{Chl}
\]

- \(\alpha_P\) : paramètre appris  
- \(\mathrm{Chl}\) : chlorophylle au **dernier pas** (mg/m³)  

#### Respiration

\[
R \;=\; \alpha_R \cdot \exp(\beta_R \cdot T) \cdot \mathrm{Chl}
\]

- \(\alpha_R,\, \beta_R\) : paramètres appris  
- \(T\) : température eau au dernier pas  
- \(\mathrm{Chl}\) : chlorophylle au dernier pas  

(La respiration est mise en forme « biomasse × facteur thermique ».)

#### Flux air–mer

\[
F
\;=\;
10^6 \cdot k_{\mathrm{gas}} \cdot \big( [\mathrm{CO}_2]_{\mathrm{eq}} - [\mathrm{CO}_2]_{\mathrm{sea}} \big)
\quad (\mu\mathrm{mol}/\mathrm{L}/\mathrm{jour})
\]

- \(k_{\mathrm{gas}} = \alpha_k \,\big( W + 0.5\,T_{\mathrm{tide}} + \epsilon \big)\), \(\alpha_k\) appris, \(\epsilon=10^{-6}\) pour éviter les divisions par zéro  
- \([\mathrm{CO}_2]_{\mathrm{eq}} = K_0 \cdot (p_{\mathrm{CO}_2}^{\mathrm{atm}} \cdot 10^{-6})\)  
- \([\mathrm{CO}_2]_{\mathrm{sea}}\) : à partir de \(\mathrm{DIC}_{\mathrm{prev}},\, [\mathrm{H}^+]_{\mathrm{prev}},\, K_1,\, K_2\) (état au **dernier pas d’entrée**)

Le facteur \(10^6\) convertit mol/L en µmol/L.

### 2.7 DIC au dernier pas et au pas prédit

- **Dernier pas d’entrée** \(t\) :  
  - \(\mathrm{pH}_{\mathrm{prev}}\) dans les entrées, \([\mathrm{H}^+]_{\mathrm{prev}} = 10^{-\mathrm{pH}_{\mathrm{prev}}}\)  
  - \(\mathrm{TA},\, K_1,\, K_2\) à \(T_{\mathrm{last}},\, S_{\mathrm{last}}\)  
  - \(\mathrm{DIC}_{\mathrm{prev}} = \mathrm{DIC}(\mathrm{TA},\, [\mathrm{H}^+]_{\mathrm{prev}},\, K_1,\, K_2)\)

- **Pas prédit** \(t+1\) :  
  - \(\mathrm{pH}_{\mathrm{pred}}\) = sortie du LSTM (prédiction de pH)  
  - \([\mathrm{H}^+]_{\mathrm{pred}} = 10^{-\mathrm{pH}_{\mathrm{pred}}}\)  
  - On garde \(\mathrm{TA},\, K_1,\, K_2\) au dernier pas (pas de T, S au pas \(t+1\))  
  - \(\mathrm{DIC}_{\mathrm{pred}} = \mathrm{DIC}(\mathrm{TA},\, [\mathrm{H}^+]_{\mathrm{pred}},\, K_1,\, K_2)\)

### 2.8 Dérivée et résidu

\[
\frac{\mathrm{d}[\mathrm{DIC}]}{\mathrm{d}t}
\;\approx\;
\mathrm{DIC}_{\mathrm{pred}} - \mathrm{DIC}_{\mathrm{prev}}
\;=\;
\Delta[\mathrm{DIC}]
\]

(Δt = 1 jour.)

\[
r
\;=\;
\Delta[\mathrm{DIC}] \;-\; (-P + R + F)
\]

### 2.9 Loss PINN

\[
\mathcal{L}_{\mathrm{pH}}
\;=\;
\lambda \cdot \frac{1}{N} \sum_{\mathrm{batch}} r^2
\]

Un résidu par échantillon (une transition par séquence), \(\lambda\) = `weight`.

### 2.10 Variables requises

| Variable   | Rôle                                                         |
|-----------|--------------------------------------------------------------|
| pH        | Entrée : dernier pas ; Cible : pas prédit (sortie LSTM)     |
| \(T\)     | Température eau (°C), dernier pas                            |
| \(S\)     | Salinité (PSS-78), dernier pas                               |
| Chl       | Chlorophylle (mg/m³), dernier pas                            |
| \(W\)     | Vent (km/h), dernier pas                                     |
| \(T_{\mathrm{tide}}\) | Marée (m), dernier pas                               |

Le **DO n’est plus utilisé** dans le pHPINN (respiration via Chl × f(T)).

### 2.11 Paramètres appris

- \(\alpha_P,\, \alpha_R,\, \beta_R,\, \alpha_k\)  
- \(p_{\mathrm{CO}_2}^{\mathrm{atm}}\) (µatm), clampé entre 200 et 600  

---

## 3. Utilisation dans le pipeline

- Les deux PINNs ajoutent une **loss physique** à la loss de régression (MSE pondérée) du LSTM.  
- Leurs **paramètres** (\(\alpha_P,\, \alpha_R,\, \ldots\)) sont inclus dans l’optimiseur avec les poids du LSTM.  
- Chaque PINN est **activé seulement** si sa variable cible (DO ou pH) est dans `target_names`.  
- Les noms des colonnes (features / targets) sont configurables dans les constructeurs (p.ex. `do_name`, `ph_name`, `temp_water_name`, etc.).

---

## 4. Références

- **Lueker et al. (2000)** — *Ocean pCO2 calculated from dissolved inorganic carbon, alkalinity, and equations for K1 and K2: validation based on laboratory measurements of CO2 in gas and seawater at equilibrium*.
- **Dickson (1990)** — *Thermodynamics of the dissociation of boric acid in synthetic seawater from 273.15 to 318.15 K*.
- **Lee et al. (2010)** — *Boron to salinity ratios for Atlantic, Arctic and Polar Waters: A view from downstream* (Ratio B/S = 0.1324).
- **Weiss (1970)** : solubilité de l’O₂ dans l’eau de mer.  
- **Weiss (1974)** : coefficient de solubilité du CO₂ (K₀).  
- **Hornbach, D.J. et al. (2021)** — *Multi-Year Monitoring of Ecosystem Metabolism in Two Branches of a Cold-Water Stream* : bilan sources–puits (photosynthèse, respiration, échange gazeux) pour l’oxygène dissous.  
- **Lee et al. (2006)** — *Global relationships of total alkalinity with salinity and temperature in surface waters of the world's oceans* : relation TA–salinité–température.  

