import pandas as pd
import numpy as np
from transformers import pipeline

# ---------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------

df = pd.read_csv("CL_F.csv")   # pétrole
df = df.dropna()

# On suppose une colonne "Close"
if "Close" not in df.columns:
    raise ValueError("Le fichier CSV doit contenir une colonne 'Close'.")

# ---------------------------------------------------------
# 2. Statistiques descriptives
# ---------------------------------------------------------

stats = df["Close"].describe()
returns = df["Close"].pct_change().dropna()

stat_text = f"""
Statistiques descriptives :
- Moyenne : {stats['mean']:.4f}
- Médiane : {stats['50%']:.4f}
- Ecart-type : {stats['std']:.4f}
- Minimum : {stats['min']:.4f}
- Maximum : {stats['max']:.4f}
- Asymétrie : {returns.skew():.4f}
- Kurtosis : {returns.kurtosis():.4f}
"""

print("\nStatistiques descriptives utilisées :")
print(stat_text)

# ---------------------------------------------------------
# 3. Chargement du modèle LLM local
# ---------------------------------------------------------
print("\nChargement du modèle LLM (open-assistant / Mistral / Llama).")

generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto"
)

# ---------------------------------------------------------
# 4. Demande 1 : Génération d'hypothèses de modèles
# ---------------------------------------------------------

prompt_hypotheses = f"""
Voici des statistiques descriptives d'une série financière :
{stat_text}

Propose des hypothèses de modèles économétriques et de deep learning 
à tester, en expliquant pour chaque modèle pourquoi il serait pertinent.
"""

response_hypotheses = generator(prompt_hypotheses, max_length=450)[0]["generated_text"]

print("\n1. Hypothèses de modèles générées par le LLM :")
print(response_hypotheses)

# ---------------------------------------------------------
# 5. Demande 2 : Rapport vulgarisé
# ---------------------------------------------------------

prompt_rapport = f"""
Voici les statistiques descriptives d'une série temporelle financière :
{stat_text}

Rédige un rapport clair, vulgarisé, compréhensible par un non-expert.
Explique ce que signifient :
- la volatilité,
- la distribution des rendements,
- les risques,
- l'incertitude,
- l'interprétation des tendances.
"""

response_rapport = generator(prompt_rapport, max_length=600)[0]["generated_text"]

print("\n2. Rapport vulgarisé généré par le LLM :")
print(response_rapport)

# ---------------------------------------------------------
# 6. Demande 3 : Recommandations d'investissement simulées
# ---------------------------------------------------------

prompt_reco = f"""
Voici les statistiques descriptives d'un actif financier :
{stat_text}

Propose des recommandations d'investissement hypothétiques 
(buy/hold/sell) basées uniquement sur cette analyse.
Explique clairement les limites, les risques, et la raison pour laquelle 
ces recommandations ne doivent pas être suivies sans analyse approfondie.
"""

response_reco = generator(prompt_reco, max_length=500)[0]["generated_text"]

print("\n3. Recommandations d'investissement simulées :")
print(response_reco)

# ---------------------------------------------------------
# 7. Comparaison humain vs IA
# ---------------------------------------------------------

explication_humaine = """
En analyse humaine classique, les statistiques montrent une volatilité 
modérée et une distribution asymétrique des rendements. Cela indique que 
l'actif peut connaître des mouvements extrêmes. L'investisseur doit rester 
prudent et ne pas baser ses décisions sur les seules statistiques passées.
"""

prompt_compare = f"""
Voici une explication humaine :
{explication_humaine}

Compare-la avec l'explication que tu as générée. 
Analyse les différences, les points communs et la cohérence globale.
"""

response_compare = generator(prompt_compare, max_length=500)[0]["generated_text"]

print("\n4. Comparaison entre explication humaine et IA :")
print(response_compare)

