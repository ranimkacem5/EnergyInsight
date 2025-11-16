# streamlit_app_complete.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import io
from openai import OpenAI
import textwrap
import base64

st.set_page_config(page_title="Analyse Pétrole & Gaz — LLM + Visuals", layout="wide")
st.title("📈 Analyse automatique — Pétrole & Gaz (Interprétation + Visuals + Export)")

from dotenv import load_dotenv
import os

# Charger les variables depuis .env

load_dotenv(dotenv_path=r"C:\Users\ranim\Desktop\projetR\.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Utilitaires ---
def is_model_results_json(data):
    if isinstance(data, dict):
        known = ["ARIMA", "SARIMA", "VAR", "GARCH", "Prophet", "LSTM", "GRU", "ARIMA+LSTM"]
        return any(k in data.keys() for k in known)
    return False

def pretty_display_json(data):
    st.json(data)

def markdown_download_button(md_text: str, filename: str):
    b = md_text.encode("utf-8")
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:file/markdown;base64,{b64}" download="{filename}">⬇️ Télécharger le rapport (.md)</a>'
    st.markdown(href, unsafe_allow_html=True)

# --- Prompt complet (amélioré : demande aussi snippets de visualisation) ---
PROMPT_TEMPLATE_WITH_VISUALS = """
Tu es un étudiant en Data Science réalisant un projet de prévision des prix du pétrole et du gaz naturel.
Tu disposes de résultats expérimentaux obtenus à partir de plusieurs modèles statistiques et de deep learning
(ARIMA, SARIMA, VAR, GARCH, Prophet+RNN, LSTM, GRU, ARIMA+LSTM).

Voici les résultats au format JSON :
{json_results}

---

### OBJECTIFS (RAPIDE)
1) Interpréter techniquement et comparer les modèles (RMSE, MAE, MAPE, AIC, BIC).
2) Générer 3 hypotheses de nouveaux modèles par actif (pétrole, gaz), justifiées.
3) Fournir des recommandations d'investissement SIMULÉES (avec limites/risques explicites).
4) Proposer 3 visualisations pertinentes et **fournir pour chaque** un snippet de code exécutable :
   - 1 snippet matplotlib pour tracé simple (prévisions vs réel),
   - 1 snippet plotly pour comparaison interactive entre modèles,
   - 1 snippet matplotlib pour diagnostics (résidus, ACF ou histogramme des erreurs).

---

### FORMAT DE SORTIE (très important)
Réponds en **Markdown** structuré :
- Partie 1 : Interprétation technique détaillée (par modèle, par actif)
- Partie 2 : Hypothèses de modèles à tester (3 par actif, justification)
- Partie 3 : Recommandation d'investissement simulée (expliciter limites et risques)
- Partie 4 : Visualisations proposées (titre, description courte, puis **bloc de code** en Python commenté pour matplotlib/plotly)
- Partie 5 : Un bref résumé vulgarisé (3-4 paragraphes maximum) pour décideur non-technique

---

### CONTRAINTES
- Les blocs de code doivent être directement exécutables si l'utilisateur a les séries 'ds' et 'y' (ou arrays de forecast).
- Sois pédagogique, explique brièvement pourquoi chaque visualisation aide à l'interprétation.

---

Fournis la sortie uniquement en MARKDOWN (avec blocs de code markdown ```python``` pour les snippets).
"""

def call_llm(json_results: dict, temperature=0.0, max_tokens=2500):
    prompt = PROMPT_TEMPLATE_WITH_VISUALS.format(
        json_results=json.dumps(json_results, indent=2, ensure_ascii=False)
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Expert en data science et finance."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        # ✔️ NOUVEAU SDK : accès correct
        return resp.choices[0].message.content

    except Exception as e:
        return f"Erreur LLM: {e}"

# --- Interface ---
uploaded = st.file_uploader("Uploader ton JSON de résultats (ou CSV pour série)", type=["json", "csv"])
col1, col2 = st.columns([2,1])

with col1:
    if uploaded:
        if uploaded.name.endswith(".json"):
            try:
                data = json.load(uploaded)
                st.success("JSON chargé.")
                pretty_display_json(data)
                if is_model_results_json(data):
                    st.markdown("### ▶️ Générer l'interprétation (LLM) avec visualisations proposées")
                    temp = st.slider("Température LLM", 0.0, 0.8, 0.0, step=0.05)
                    if st.button("Générer rapport & snippets"):
                        with st.spinner("Appel LLM en cours..."):
                            report_md = call_llm(data, temperature=float(temp))
                        st.subheader("📝 Rapport (Markdown généré par LLM)")
                        st.markdown(report_md, unsafe_allow_html=True)
                        # option download .md
                        markdown_download_button(report_md, "rapport_interpretation.md")
                else:
                    st.info("Le JSON ne semble pas contenir des résultats de modèles reconnus. Si c'est une série temporelle (CSV), uploade un CSV.")
            except Exception as e:
                st.error(f"Impossible de parser le JSON : {e}")
        else:
            # CSV -> afficher série et proposer d'extraire features
            try:
                df = pd.read_csv(uploaded)
                st.subheader("CSV détecté — aperçu")
                st.dataframe(df.head())
                # détecte colonne date probable
                date_cols = [c for c in df.columns if "date" in c.lower() or c.lower() == "ds"]
                if date_cols:
                    st.success(f"Colonne date détectée : {date_cols[0]}")
                else:
                    st.warning("Aucune colonne date détectée automatiquement (reformatez votre CSV avec une colonne 'date' ou 'ds').")
            except Exception as e:
                st.error(f"Erreur lecture CSV : {e}")

with col2:
    st.markdown("## 🧰 Outils")
    st.markdown("- Exemple JSON de test disponible dans l'interface (copier & coller).")
    if st.button("Afficher exemple JSON"):
        st.code(open(__file__).read() if "__file__" in globals() else "Voir l'exemple JSON fourni séparément.", language="python")
    st.markdown("---")
    st.markdown("## 🔒 Sécurité")
    st.markdown("- Ne partage jamais ta clé dans un repo public.")
    st.markdown("- Le script sauvegarde uniquement côté client la réponse Markdown via téléchargement.")
    st.markdown("---")
    st.markdown("## ✅ Prochaines améliorations possibles")
    st.markdown("""
    - Ajouter exécution automatique des snippets (si on fournit les séries 'ds','y' et forecasts).
    - Export PDF via wkhtmltopdf / pandoc (nécessite dépendances serveur).
    - Intégrer une option 'Run Visuals' qui exécute les snippets et montre les figures dans Streamlit.
    """)

st.markdown("---")
st.caption("Remplace `VOTRE_CLE_API_ICI` par ta clé locale. Les appels LLM peuvent coûter selon ton plan.")
