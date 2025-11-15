# ==========================================================
# Fichier : export_results.R
# Objectif : Exporter les résultats des modèles classiques R
# ==========================================================

# Charger les bibliothèques
library(jsonlite)
library(Metrics)

# ⚠️ Avant d'exécuter ce script, assure-toi que :
#  - Tu as déjà exécuté 03_modeles_classiques.R
#  - Les objets suivants existent :
#    rmse_pet, mae_pet, rmse_pet_sarima, mae_pet_sarima, var_model, garch_pet, garch_gaz, etc.

# ----------------------------------------------------------
# 1️⃣ Résultats ARIMA
# ----------------------------------------------------------
arima_results <- list(
  model = "ARIMA",
  dataset = "Pétrole (CL=F)",
  metrics = list(
    RMSE = round(rmse_pet, 4),
    MAE  = round(mae_pet, 4)
  ),
  observations = "Le modèle ARIMA capte bien la tendance principale du pétrole, mais ne prend pas en compte la volatilité.",
  date = as.character(Sys.Date())
)

# ----------------------------------------------------------
# 2️⃣ Résultats SARIMA
# ----------------------------------------------------------
sarima_results <- list(
  model = "SARIMA",
  dataset = "Pétrole (CL=F)",
  metrics = list(
    RMSE = round(rmse_pet_sarima, 4),
    MAE  = round(mae_pet_sarima, 4)
  ),
  observations = "SARIMA améliore légèrement les prévisions grâce à la composante saisonnière.",
  date = as.character(Sys.Date())
)

# ----------------------------------------------------------
# 3️⃣ Résultats VAR
# ----------------------------------------------------------
var_rmse <- rmse(petrole_test_df$CL.F.Close[1:n_forecast], forecast_pet_var)
var_mae  <- mae(petrole_test_df$CL.F.Close[1:n_forecast], forecast_pet_var)

var_results <- list(
  model = "VAR",
  dataset = "Gaz et Pétrole",
  metrics = list(
    RMSE = round(var_rmse, 4),
    MAE  = round(var_mae, 4)
  ),
  observations = "VAR capture la relation entre les prix du pétrole et du gaz naturel.",
  date = as.character(Sys.Date())
)

# ----------------------------------------------------------
# 4️⃣ Résultats GARCH
# ----------------------------------------------------------
garch_results <- list(
  model = "GARCH(1,1)",
  dataset = "Pétrole (CL=F)",
  metrics = list(
    AIC = round(infocriteria(garch_pet)[1], 4),
    BIC = round(infocriteria(garch_pet)[2], 4)
  ),
  observations = "Le modèle GARCH modélise efficacement la volatilité du pétrole.",
  date = as.character(Sys.Date())
)

# ----------------------------------------------------------
# 5️⃣ Résultats ARCH
# ----------------------------------------------------------
arch_results <- list(
  model = "ARCH(1)",
  dataset = "Gaz (UNG)",
  metrics = list(
    AIC = round(infocriteria(garch_gaz_arch)[1], 4),
    BIC = round(infocriteria(garch_gaz_arch)[2], 4)
  ),
  observations = "Le modèle ARCH capture la volatilité ponctuelle du gaz.",
  date = as.character(Sys.Date())
)

# ----------------------------------------------------------
# 6️⃣ Créer la structure finale et sauvegarder en JSON
# ----------------------------------------------------------
results_list <- list(
  ARIMA  = arima_results,
  SARIMA = sarima_results,
  VAR    = var_results,
  GARCH  = garch_results,
  ARCH   = arch_results
)

# Créer le dossier s’il n’existe pas
if(!dir.exists("reports")) dir.create("reports", recursive = TRUE)

# Sauvegarde du fichier JSON
write_json(results_list, "reports/results_R_models.json", pretty = TRUE)

cat("✅ Fichier exporté avec succès : reports/results_R_models.json\n")
