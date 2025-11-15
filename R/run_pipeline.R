# ==========================================================
# 🚀 Fichier : run_pipeline.R
# Objectif : Exécuter automatiquement toutes les étapes du projet R
# ==========================================================

# -----------------------------
# 🔧 Étape 0 : Préparation
# -----------------------------
cat("=====================================\n")
cat("  🚀 DÉMARRAGE DU PIPELINE R\n")
cat("=====================================\n\n")

# Création des dossiers nécessaires
dirs <- c("data", "reports", "logs")
for (d in dirs) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

# Début du fichier de log
log_file <- "logs/pipeline_log.txt"
sink(log_file, append = TRUE)
cat("\n=====================================\n")
cat("Lancement du pipeline - ", Sys.time(), "\n")
cat("=====================================\n")

# Fonction helper pour exécuter chaque étape avec gestion d’erreur
run_step <- function(script_path, step_name) {
  cat(paste0("\n➡️  Étape : ", step_name, "...\n"))
  tryCatch({
    source(script_path)
    cat(paste0("✅ Étape réussie : ", step_name, "\n"))
  }, error = function(e) {
    cat(paste0("❌ ERREUR dans ", step_name, " : ", e$message, "\n"))
  })
}

# -----------------------------
# 1️⃣ Préparation des données
# -----------------------------
run_step("R/01_preparation_data.R", "Préparation des données")

# -----------------------------
# 2️⃣ Analyse exploratoire
# -----------------------------
run_step("R/02_analyse_exploratoire.R", "Analyse exploratoire")

# -----------------------------
# 3️⃣ Modèles classiques (ARIMA, SARIMA, VAR, GARCH, ARCH)
# -----------------------------
run_step("R/03_modeles_classiques.R", "Modélisation classique")

# -----------------------------
# 4️⃣ Export des résultats (JSON)
# -----------------------------
run_step("R/export_results.R", "Export des résultats")

# -----------------------------
# ✅ Fin du pipeline
# -----------------------------
cat("\n=====================================\n")
cat("🎉 PIPELINE TERMINÉ AVEC SUCCÈS - ", Sys.time(), "\n")
cat("Résultats disponibles dans : reports/results_R_models.json\n")
cat("Logs enregistrés dans : logs/pipeline_log.txt\n")
cat("=====================================\n")

# Fermer le flux du log
sink()
