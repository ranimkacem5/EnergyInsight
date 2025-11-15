# ==========================================================
# 🚀 Fichier : run_pipeline.R (simplifié)
# Objectif : Exécuter automatiquement toutes les étapes du projet R
# ==========================================================

library(jsonlite)

# -----------------------------
# 🔧 Paramètres et chemins
# -----------------------------
dirs <- c("data", "reports", "logs", "R")
log_file <- "logs/pipeline_log.txt"
error_file <- "logs/errors.json"
result_file <- "reports/model_results.json"

scripts <- list(
  "Préparation des données" = "R/01_preparation_data.R",
  "Analyse exploratoire"   = "R/02_analyse_exploratoire.R",
  "Modélisation classique"  = "R/03_modeles_classiques.R"
)

# Création des dossiers si inexistants
for (d in dirs) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

# Initialisation du log
sink(log_file, append = TRUE)
cat("\n=====================================\n")
cat("🚀 LANCEMENT DU PIPELINE - ", Sys.time(), "\n")
cat("=====================================\n")

# Liste pour stocker les erreurs
errors <- list()

# -----------------------------
# 🔹 Fonction helper
# -----------------------------
run_step <- function(script_path, step_name) {
  cat(paste0("\n➡️  Étape : ", step_name, "...\n"))
  start_time <- Sys.time()
  
  if (!file.exists(script_path)) {
    msg <- paste0("❌ Script introuvable : ", script_path)
    cat(msg, "\n")
    errors[[step_name]] <<- msg
    return(NULL)
  }
  
  tryCatch({
    source(script_path)
    duration <- round(difftime(Sys.time(), start_time, units = "secs"), 2)
    cat(paste0("✅ Étape réussie : ", step_name, " (", duration, " sec)\n"))
  }, error = function(e) {
    msg <- paste0("❌ ERREUR dans ", step_name, " : ", e$message)
    cat(msg, "\n")
    errors[[step_name]] <<- e$message
  })
}

# -----------------------------
# 🔹 Exécution des scripts
# -----------------------------
for (step in names(scripts)) {
  run_step(scripts[[step]], step)
}

# -----------------------------
# 🔹 Sauvegarde des erreurs (si présentes)
# -----------------------------
if (length(errors) > 0) {
  write_json(errors, error_file, pretty = TRUE, auto_unbox = TRUE)
  cat("\n⚠️ Certaines étapes ont échoué. Voir :", error_file, "\n")
}

# -----------------------------
# ✅ Fin du pipeline
# -----------------------------
cat("\n=====================================\n")
cat("🎉 PIPELINE TERMINÉ - ", Sys.time(), "\n")
cat("Résultats disponibles dans :", result_file, "\n")
cat("Logs enregistrés dans :", log_file, "\n")
cat("=====================================\n")
sink()

