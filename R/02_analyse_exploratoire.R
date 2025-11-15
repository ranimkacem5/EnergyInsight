# ==========================================================
# 02_analyse_exploratoire.R
# Objectif : Analyse visuelle et statistique des séries
# ==========================================================

library(ggplot2)
library(gridExtra)
library(zoo)
library(tseries)
library(forecast)

# Charger les données préparées
load("data/splits.RData")

# Conversion en data.frame pour visualisation
petrole_train_df <- data.frame(date = index(petrole_split$train), coredata(petrole_split$train))
gaz_train_df <- data.frame(date = index(gaz_split$train), coredata(gaz_split$train))

# ---- Évolution temporelle ----
p1 <- ggplot(petrole_train_df, aes(x = date, y = CL.F.Close)) +
  geom_line(color = "steelblue") +
  labs(title = "Évolution du prix du pétrole (WTI)", x = "Date", y = "Prix (USD)") +
  theme_minimal()

p2 <- ggplot(gaz_train_df, aes(x = date, y = UNG.Close)) +
  geom_line(color = "darkgreen") +
  labs(title = "Évolution du prix du gaz naturel (UNG)", x = "Date", y = "Prix (USD)") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 1)

# ---- Moyenne mobile ----
petrole_train_df$MA30 <- zoo::rollmean(petrole_train_df$CL.F.Close, k = 30, fill = NA)
gaz_train_df$MA30 <- zoo::rollmean(gaz_train_df$UNG.Close, k = 30, fill = NA)

ggplot(petrole_train_df, aes(x = date)) +
  geom_line(aes(y = CL.F.Close), color = "steelblue", alpha = 0.6) +
  geom_line(aes(y = MA30), color = "red", size = 1) +
  labs(title = "Tendance du pétrole (moyenne mobile 30 jours)") +
  theme_minimal()

# ---- Stationnarité ----
adf_pet <- adf.test(petrole_train_df$CL.F.Close)
adf_gaz <- adf.test(gaz_train_df$UNG.Close)

cat("ADF test - Pétrole :", adf_pet$p.value, "\n")
cat("ADF test - Gaz :", adf_gaz$p.value, "\n")

# ---- ACF / PACF ----
par(mfrow = c(2, 2))
acf(diff(petrole_train_df$CL.F.Close), main = "ACF - Pétrole (diff)")
pacf(diff(petrole_train_df$CL.F.Close), main = "PACF - Pétrole (diff)")
acf(diff(gaz_train_df$UNG.Close), main = "ACF - Gaz (diff)")
pacf(diff(gaz_train_df$UNG.Close), main = "PACF - Gaz (diff)")
par(mfrow = c(1, 1))

# ---- Boxplots ----
par(mfrow = c(2, 3))
for (col in colnames(petrole_split$train)) {
  boxplot(petrole_split$train[, col], main = paste("Boxplot -", col), col = "skyblue")
}
par(mfrow = c(1, 1))

