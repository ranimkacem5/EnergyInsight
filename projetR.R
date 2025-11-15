# Charger les packages nécessaires
library(quantmod)
library(tidyquant)
library(forecast)
library(tseries)
library(vars)
library(ggplot2)
library(xts)
library(zoo)
library(gridExtra)
library(VIM)
library(DMwR2)
install.packages("tseries")
install.packages("Metrics")
library(Metrics)
library(forecast)
library(tseries)
install.packages("xts")   # si pas encore installé
library(xts)               # indispensable avant utilisation



install.packages("quantmod")
library(quantmod)

# Définir la période
start_date <- as.Date("2015-01-01")
end_date <- Sys.Date()

# Pétrole WTI
getSymbols("CL=F", src="yahoo", from=start_date, to=end_date)
petrole <- data.frame(date=index(`CL=F`), coredata(`CL=F`))

# Gaz naturel via ETF UNG
getSymbols("UNG", src="yahoo", from=start_date, to=end_date)
gaz <- data.frame(date=index(UNG), coredata(UNG))

write.csv(petrole, "C:/Users/ranim/Desktop/projetR/CL_F.csv", row.names = FALSE)
write.csv(gaz, "C:/Users/ranim/Desktop/projetR/UNG.csv", row.names = FALSE)

petrole <- read.csv("CL_F.csv")
gaz <- read.csv("UNG.csv")



# Voir la structure
str(petrole)
str(gaz)

# Résumé statistique
summary(petrole)
summary(gaz)


# Nombre de valeurs manquantes par colonne
colSums(is.na(petrole))
colSums(is.na(gaz))

install.packages(c("VIM", "DMwR2")) # Pour imputation KNN
library(VIM)
library(DMwR2)




cat("Valeurs manquantes PETROLE :\n")
print(colSums(is.na(petrole)))

# Appliquer KNN (k=5 voisins)
petrole[, 2:7] <- kNN(petrole[, 2:7], k=5, imp_var=FALSE)
# Vérifier s'il reste des NA
colSums(is.na(petrole))

# Afficher les valeurs manquantes avant KNN
cat("Valeurs manquantes GAZ avant KNN :\n")
print(colSums(is.na(gaz)))

# Appliquer KNN (k=5 voisins) sur les colonnes numériques
gaz[, 2:7] <- kNN(gaz[, 2:7], k=5, imp_var=FALSE)

# Vérifier s'il reste des NA après KNN
cat("Valeurs manquantes GAZ après KNN :\n")
print(colSums(is.na(gaz)))

library(xts)

# Transformer petrole en série temporelle multidimensionnelle
petrole_xts <- xts(petrole[, 2:7], order.by=as.Date(petrole$date))
gaz_xts <- xts(petrole[, 2:7], order.by=as.Date(gaz$date))

# ----- TRANSFORMATION EN TRAIN / TEST -----
# Pétrole
n_pet <- nrow(petrole_xts)
train_size_pet <- floor(0.9 * n_pet)
petrole_train <- petrole_xts[1:train_size_pet, ]
petrole_test  <- petrole_xts[(train_size_pet+1):n_pet, ]

# Gaz
n_gaz <- nrow(gaz_xts)
train_size_gaz <- floor(0.9 * n_gaz)
gaz_train <- gaz_xts[1:train_size_gaz, ]
gaz_test  <- gaz_xts[(train_size_gaz+1):n_gaz, ]

# ----- ANALYSE EXPLORATOIRE SUR LE TRAIN -----
petrole_train_df <- data.frame(
  date = index(petrole_train),
  coredata(petrole_train)
)
# Charger ggplot2
library(ggplot2)

# Si tu veux afficher plusieurs graphiques côte à côte
library(gridExtra)

gaz_train_df <- data.frame(
  date = index(gaz_train),
  coredata(gaz_train)
)
p1 <- ggplot(petrole_train_df, aes(x = date, y = CL.F.Close)) +
  geom_line(color = "steelblue") +
  labs(title = "Évolution du prix du pétrole (WTI) - Train",
       x = "Date", y = "Prix de clôture (USD)") +
  theme_minimal()


p2 <- ggplot(as.data.frame(gaz_train), aes(x = index(gaz_train), y = gaz_train$UNG.Close)) +
  geom_line(color = "darkgreen") +
  labs(title = "Évolution du prix du gaz naturel (UNG) - Train",
       x = "Date", y = "Prix de clôture (USD)") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 1)

# 2️⃣ Moyenne mobile pour visualiser la tendance
petrole_train_df <- as.data.frame(petrole_train)
gaz_train_df <- as.data.frame(gaz_train)

petrole_train_df$MA30 <- zoo::rollmean(petrole_train_df$CL.F.Close, k = 30, fill = NA)
gaz_train_df$MA30 <- zoo::rollmean(gaz_train_df$UNG.Close, k = 30, fill = NA)

ggplot(petrole_train_df, aes(x = index(petrole_train))) +
  geom_line(aes(y = CL.F.Close), color = "steelblue", alpha = 0.6) +
  geom_line(aes(y = MA30), color = "red", size = 1) +
  labs(title = "Pétrole WTI - Tendance (moyenne mobile 30 jours) - Train",
       x = "Date", y = "Prix") +
  theme_minimal()

ggplot(gaz_train_df, aes(x = index(gaz_train))) +
  geom_line(aes(y = UNG.Close), color = "darkgreen", alpha = 0.6) +
  geom_line(aes(y = MA30), color = "red", size = 1) +
  labs(title = "Gaz Naturel - Tendance (moyenne mobile 30 jours) - Train",
       x = "Date", y = "Prix") +
  theme_minimal()

# Décomposition du pétrole
petrole_decomp <- decompose(petrole_train_ts)
plot(petrole_decomp)  # ne pas mettre main

# Décomposition du gaz
gaz_decomp <- decompose(gaz_train_ts)
plot(gaz_decomp)      # ne pas mettre main

# Pétrole
plot(petrole_decomp$trend, main="Tendance - Pétrole (Train)")
plot(petrole_decomp$seasonal, main="Saisonnalité - Pétrole (Train)")
plot(petrole_decomp$random, main="Résidu - Pétrole (Train)")

# Gaz
plot(gaz_decomp$trend, main="Tendance - Gaz (Train)")
plot(gaz_decomp$seasonal, main="Saisonnalité - Gaz (Train)")
plot(gaz_decomp$random, main="Résidu - Gaz (Train)")

# 4️⃣ Stationnarité et différenciation
adf.test(petrole_train_df$CL.F.Close)
petrole_diff <- diff(petrole_train_df$CL.F.Close)
adf.test(na.omit(petrole_diff))

adf.test(gaz_train_df$UNG.Close)
gaz_diff <- diff(gaz_train_df$UNG.Close)
adf.test(na.omit(gaz_diff))

# 5️⃣ ACF / PACF pour le train
par(mfrow = c(2,1))
acf(na.omit(petrole_diff), main = "ACF - Pétrole différencié (Train)")
pacf(na.omit(petrole_diff), main = "PACF - Pétrole différencié (Train)")

acf(na.omit(gaz_diff), main = "ACF - Gaz différencié (Train)")
pacf(na.omit(gaz_diff), main = "PACF - Gaz différencié (Train)")
par(mfrow = c(1,1))

# 6️⃣ Boxplots pour toutes les variables du train
vars_pet_train <- colnames(petrole_train)
vars_gaz_train <- colnames(gaz_train)

par(mfrow = c(2,3))
for(col in vars_pet_train){
  boxplot(petrole_train[,col], main = paste("Boxplot -", col, "(Train)"), col="skyblue")
}
par(mfrow = c(1,1))

par(mfrow = c(2,3))
for(col in vars_gaz_train){
  boxplot(gaz_train[,col], main = paste("Boxplot -", col, "(Train)"), col="lightgreen")
}
par(mfrow = c(1,1))


# Pétrole train/test en data.frame
petrole_train_df <- data.frame(date=index(petrole_train), CL.F.Close=coredata(petrole_train)[, "CL.F.Close"])
petrole_test_df  <- data.frame(date=index(petrole_test), CL.F.Close=coredata(petrole_test)[, "CL.F.Close"])

gaz_train_df <- data.frame(date=index(gaz_train), UNG.Close=coredata(gaz_train)[, "UNG.Close"])
gaz_test_df  <- data.frame(date=index(gaz_test), UNG.Close=coredata(gaz_test)[, "UNG.Close"])

# ------------------------ 3️⃣ Ajuster ARIMA sur le train ------------------------
# Pétrole
auto_arima_pet <- auto.arima(
  petrole_train_df$CL.F.Close,
  seasonal=FALSE,         # pas de saisonnalité mensuelle/annuelle
  stepwise=FALSE,         # recherche exhaustive
  approximation=FALSE
)
summary(auto_arima_pet)

# Gaz
auto_arima_gaz <- auto.arima(
  gaz_train_df$UNG.Close,
  seasonal=FALSE,
  stepwise=FALSE,
  approximation=FALSE
)
summary(auto_arima_gaz)

# ------------------------ 4️⃣ Générer les prévisions sur le test ------------------------
forecast_pet <- forecast(auto_arima_pet, h = nrow(petrole_test_df))
forecast_gaz <- forecast(auto_arima_gaz, h = nrow(gaz_test_df))

# ------------------------ 5️⃣ Évaluer les performances ------------------------
# Pétrole
rmse_pet <- rmse(petrole_test_df$CL.F.Close, as.numeric(forecast_pet$mean))
mae_pet  <- mae(petrole_test_df$CL.F.Close, as.numeric(forecast_pet$mean))
cat("Pétrole - RMSE:", rmse_pet, "MAE:", mae_pet, "\n")

# Gaz
rmse_gaz <- rmse(gaz_test_df$UNG.Close, as.numeric(forecast_gaz$mean))
mae_gaz  <- mae(gaz_test_df$UNG.Close, as.numeric(forecast_gaz$mean))
cat("Gaz - RMSE:", rmse_gaz, "MAE:", mae_gaz, "\n")




# ------------------------ 1️⃣ Ajuster SARIMA sur le train ------------------------
# Pétrole
sarima_pet <- auto.arima(
  petrole_train_df$CL.F.Close,
  seasonal = TRUE,       # détecte la saisonnalité automatiquement
  stepwise = FALSE,
  approximation = FALSE
)
summary(sarima_pet)

# Gaz
sarima_gaz <- auto.arima(
  gaz_train_df$UNG.Close,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE
)
summary(sarima_gaz)

# ------------------------ 2️⃣ Générer les prévisions sur le test ------------------------
forecast_pet_sarima <- forecast(sarima_pet, h = nrow(petrole_test_df))
forecast_gaz_sarima <- forecast(sarima_gaz, h = nrow(gaz_test_df))

# ------------------------ 3️⃣ Calculer RMSE / MAE ------------------------
# Pétrole
rmse_pet_sarima <- rmse(petrole_test_df$CL.F.Close, as.numeric(forecast_pet_sarima$mean))
mae_pet_sarima  <- mae(petrole_test_df$CL.F.Close, as.numeric(forecast_pet_sarima$mean))
cat("Pétrole SARIMA - RMSE:", rmse_pet_sarima, "MAE:", mae_pet_sarima, "\n")

# Gaz
rmse_gaz_sarima <- rmse(gaz_test_df$UNG.Close, as.numeric(forecast_gaz_sarima$mean))
mae_gaz_sarima  <- mae(gaz_test_df$UNG.Close, as.numeric(forecast_gaz_sarima$mean))
cat("Gaz SARIMA - RMSE:", rmse_gaz_sarima, "MAE:", mae_gaz_sarima, "\n")



rm(list=ls())
library(vars)


# Aligner les séries et créer un data.frame
min_rows <- min(nrow(petrole_train_df), nrow(gaz_train_df))
var_train_df <- data.frame(
  petrole = petrole_train_df$CL.F.Close[1:min_rows],
  gaz = gaz_train_df$UNG.Close[1:min_rows]
)

# Vérifier stationnarité et différencier si nécessaire
var_train_diff <- diff(as.matrix(var_train_df))
var_train_diff <- na.omit(var_train_diff)

# Sélection du lag optimal
lag_selection <- VARselect(var_train_diff, lag.max=10, type="const")
p <- lag_selection$selection["AIC(n)"]

# Ajustement du VAR
var_model <- VAR(var_train_diff, p=p, type="const")

# Prévisions
n_forecast <- min(nrow(petrole_test_df), nrow(gaz_test_df))
var_forecast <- predict(var_model, n.ahead=n_forecast)

forecast_pet_var <- var_forecast$fcst$petrole[, "fcst"]
forecast_gaz_var <- var_forecast$fcst$gaz[, "fcst"]

# RMSE / MAE
rmse(petrole_test_df$CL.F.Close[1:n_forecast], forecast_pet_var)
mae(petrole_test_df$CL.F.Close[1:n_forecast], forecast_pet_var)
rmse(gaz_test_df$UNG.Close[1:n_forecast], forecast_gaz_var)
mae(gaz_test_df$UNG.Close[1:n_forecast], forecast_gaz_var)

-----------------------grach----------------------------------------
  install.packages("rugarch")
library(rugarch)
# Retours log pour stationnarité
log_return_pet <- diff(log(petrole_train_df$CL.F.Close))
log_return_gaz <- diff(log(gaz_train_df$UNG.Close))

# Supprimer les NA générés par la différence
log_return_pet <- na.omit(log_return_pet)
log_return_gaz <- na.omit(log_return_gaz)

# Pétrole
spec_pet <- ugarchspec(
  variance.model = list(model="sGARCH", garchOrder=c(1,1)),
  mean.model = list(armaOrder=c(1,1), include.mean=TRUE),
  distribution.model = "norm"
)

garch_pet <- ugarchfit(spec=spec_pet, data=log_return_pet)
summary(garch_pet)

# Gaz
spec_gaz <- ugarchspec(
  variance.model = list(model="sGARCH", garchOrder=c(1,1)),
  mean.model = list(armaOrder=c(1,1), include.mean=TRUE),
  distribution.model = "norm"
)

garch_gaz <- ugarchfit(spec=spec_gaz, data=log_return_gaz)
summary(garch_gaz)
# Nombre de jours à prévoir
n_forecast_pet <- nrow(petrole_test_df)
n_forecast_gaz <- nrow(gaz_test_df)

# Prévision pour le pétrole
garch_pet_forecast <- ugarchforecast(garch_pet, n.ahead=n_forecast_pet)
sigma_pet_forecast <- sigma(garch_pet_forecast)

# Prévision pour le gaz
garch_gaz_forecast <- ugarchforecast(garch_gaz, n.ahead=n_forecast_gaz)
sigma_gaz_forecast <- sigma(garch_gaz_forecast)
# Pétrole
plot(sigma_pet_forecast, type="l", main="Volatilité prévisionnelle - Pétrole WTI", ylab="Volatilité", xlab="Jours")

# Gaz
plot(sigma_gaz_forecast, type="l", main="Volatilité prévisionnelle - Gaz Naturel", ylab="Volatilité", xlab="Jours")

-----------------------------arch---------------------------------------------------
  log_return_pet <- na.omit(diff(log(petrole_train_df$CL.F.Close)))
log_return_gaz <- na.omit(diff(log(gaz_train_df$UNG.Close)))

# ------------------------ 2️⃣ Spécification du modèle ARCH ------------------------


# Pétrole
spec_pet_arch <- ugarchspec(
  variance.model = list(model="sGARCH", garchOrder=c(1,0)),  # ARCH(1)
  mean.model = list(armaOrder=c(1,1), include.mean=TRUE),
  distribution.model = "norm"
)

# Gaz
spec_gaz_arch <- ugarchspec(
  variance.model = list(model="sGARCH", garchOrder=c(1,0)),  # ARCH(1)
  mean.model = list(armaOrder=c(1,1), include.mean=TRUE),
  distribution.model = "norm"
)

# ------------------------ 3️⃣ Ajustement du modèle ------------------------
# Pétrole
garch_pet_arch <- ugarchfit(spec=spec_pet_arch, data=log_return_pet)
summary(garch_pet_arch)

# Gaz
garch_gaz_arch <- ugarchfit(spec=spec_gaz_arch, data=log_return_gaz)
summary(garch_gaz_arch)

# ------------------------ 4️⃣ Prévision de la volatilité ------------------------
# Définir horizon de prévision (nombre de jours dans le test)
n_forecast_pet <- nrow(petrole_test_df)
n_forecast_gaz <- nrow(gaz_test_df)

# Prévision pour le pétrole
arch_pet_forecast <- ugarchforecast(garch_pet_arch, n.ahead=n_forecast_pet)
sigma_pet_forecast <- sigma(arch_pet_forecast)  # volatilité prévisionnelle

# Prévision pour le gaz
arch_gaz_forecast <- ugarchforecast(garch_gaz_arch, n.ahead=n_forecast_gaz)
sigma_gaz_forecast <- sigma(arch_gaz_forecast)

# ------------------------ 5️⃣ Visualisation ------------------------
# Pétrole
plot(sigma_pet_forecast, type="l",
     main="Volatilité prévisionnelle - Pétrole WTI (ARCH)",
     ylab="Volatilité", xlab="Jours")

# Gaz
plot(sigma_gaz_forecast, type="l",
     main="Volatilité prévisionnelle - Gaz Naturel (ARCH)",
     ylab="Volatilité", xlab="Jours")

# ------------------------ 6️⃣ Optionnel : Intégration avec test ------------------------
# Si tu veux comparer la volatilité prévisionnelle avec la volatilité réelle :
real_vol_pet <- abs(diff(log(petrole_test_df$CL.F.Close)))
real_vol_gaz <- abs(diff(log(gaz_test_df$UNG.Close)))

# Tracer overlay (optionnel)
plot(real_vol_pet, type="l", col="blue", main="Volatilité réelle vs ARCH - Pétrole",
     ylab="Volatilité", xlab="Jours")
lines(sigma_pet_forecast, col="red")
legend("topright", legend=c("Réelle","Prévision ARCH"), col=c("blue","red"), lty=1)

#Modélisation classique
# ARIMA résidus
checkresiduals(auto_arima_pet)
checkresiduals(auto_arima_gaz)

# SARIMA résidus
checkresiduals(sarima_pet)
checkresiduals(sarima_gaz)

# VAR résidus
serial.test(var_model, lags.pt=10, type="PT.asymptotic")




