# ==========================================================
# 01_preparation_data.R
# Objectif : Collecte, nettoyage et préparation des données
# ==========================================================

# 📦 Packages nécessaires
library(quantmod)
library(VIM)
library(DMwR2)
library(xts)
library(tidyquant)

# 📅 Définir la période d'étude
start_date <- as.Date("2015-01-01")
end_date <- Sys.Date()

# 📈 Télécharger les données Yahoo Finance
getSymbols("CL=F", src="yahoo", from=start_date, to=end_date)   # Pétrole WTI
getSymbols("UNG", src="yahoo", from=start_date, to=end_date)   # Gaz naturel via ETF UNG

# 🔄 Conversion en data.frame
petrole <- data.frame(date = index(`CL=F`), coredata(`CL=F`))
gaz <- data.frame(date = index(UNG), coredata(UNG))

# 💾 Sauvegarde brute (si besoin)
write.csv(petrole, "data/CL_F.csv", row.names = FALSE)
write.csv(gaz, "data/UNG.csv", row.names = FALSE)

# 🧼 Gestion des valeurs manquantes
cat("Valeurs manquantes - Pétrole :\n")
print(colSums(is.na(petrole)))

cat("Valeurs manquantes - Gaz :\n")
print(colSums(is.na(gaz)))

# Imputation KNN pour compléter les valeurs manquantes
petrole[, 2:7] <- kNN(petrole[, 2:7], k = 5, imp_var = FALSE)
gaz[, 2:7] <- kNN(gaz[, 2:7], k = 5, imp_var = FALSE)

# ✅ Vérification après imputation
cat("Valeurs manquantes après imputation (Pétrole):\n")
print(colSums(is.na(petrole)))

cat("Valeurs manquantes après imputation (Gaz):\n")
print(colSums(is.na(gaz)))

# 🔁 Conversion en séries temporelles multivariées
petrole_xts <- xts(petrole[, 2:7], order.by = as.Date(petrole$date))
gaz_xts <- xts(gaz[, 2:7], order.by = as.Date(gaz$date))

# 🔀 Split Train/Test (90% / 10%)
split_xts <- function(data_xts) {
  n <- nrow(data_xts)
  train_size <- floor(0.9 * n)
  list(
    train = data_xts[1:train_size, ],
    test  = data_xts[(train_size + 1):n, ]
  )
}

petrole_split <- split_xts(petrole_xts)
gaz_split <- split_xts(gaz_xts)

# 💾 Sauvegarde pour étapes suivantes
save(petrole_split, gaz_split, file = "data/splits.RData")

cat("✅ Données prêtes et sauvegardées dans data/splits.RData\n")

