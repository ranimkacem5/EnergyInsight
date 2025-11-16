🧾 Description du projet
Titre du projet :

Analyse et Prévision des Prix du Pétrole et du Gaz à l’aide de Modèles Statistiques, Deep Learning et LLM

🧠 Contexte

Les fluctuations des prix du pétrole et du gaz influencent directement l’économie mondiale, les décisions d’investissement et les politiques énergétiques.
Ce projet vise à modéliser, prédire et interpréter ces variations à l’aide de techniques avancées de Data Science, combinant à la fois des modèles classiques de séries temporelles et des approches deep learning modernes.

🎯 Objectifs du projet

Analyser l’évolution historique des prix du pétrole et du gaz.

Appliquer différents modèles de prévision : ARIMA, SARIMA, VAR, GARCH, LSTM, GRU, Prophet+RNN, ARIMA+LSTM.

Comparer leurs performances selon les métriques RMSE, MAE, MAPE.

Générer automatiquement une interprétation des résultats à l’aide d’un LLM (Large Language Model).

Produire des hypothèses de nouveaux modèles à tester à partir des statistiques descriptives.

Fournir des recommandations simulées d’investissement accompagnées d’une explication des limites et risques.

🧩 Méthodologie

Préparation des données : nettoyage, différenciation, et normalisation des séries temporelles.

Application des modèles statistiques : ARIMA, SARIMA, VAR, GARCH.

Implémentation des modèles neuronaux : LSTM, GRU, Prophet+RNN, hybrides ARIMA+LSTM.

Évaluation des performances à l’aide des indicateurs RMSE, MAE et MAPE.

Interprétation automatisée via un prompt LLM générant :

Analyse technique des modèles

Hypothèses de modèles à tester

Rapport vulgarisé et recommandations d’investissement.

🧰 Technologies utilisées

Langages : Python

Bibliothèques principales : statsmodels, arch, prophet, tensorflow / keras, numpy, pandas, matplotlib

LLM : GPT-5 (utilisé pour l’interprétation automatique des résultats et la génération d’hypothèses)

Évaluation : RMSE, MAE, MAPE

Visualisation : Matplotlib, Seaborn

📊 Résultats

Les modèles hybrides Prophet+RNN et ARIMA+LSTM présentent les meilleures performances (RMSE ≈ 1.45, MAE ≈ 1.11).

Les modèles classiques (ARIMA, SARIMA) donnent de bons résultats pour le pétrole mais sont moins efficaces sur le gaz à forte volatilité.

Le LLM permet de générer automatiquement un rapport explicatif, vulgariser les résultats et proposer des modèles alternatifs de manière cohérente.

📈 Perspectives

Amélioration de la robustesse via des modèles hybrides LSTM+GARCH.

Intégration de facteurs externes (géopolitiques, macroéconomiques).

Automatisation complète du cycle d’analyse via une pipeline IA explicable
