# Installer les packages si nécessaire
install.packages(c("quantmod", "tidyquant", "forecast", "tseries", "vars", "ggplot2", "xts", "keras", "abind", "rugarch"))

# Charger les packages
library(quantmod)
library(tidyquant)
library(forecast)
library(tseries)
library(vars)
library(ggplot2)
library(xts)
library(keras)
library(abind)
library(rugarch)

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

# Afficher les premières lignes
head(petrole)
head(gaz)

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

# Vérifier
summary(petrole)

 
cat("Valeurs manquantes PETROLE :\n")
print(colSums(is.na(petrole)))
 
# Appliquer KNN (k=5 voisins)
petrole[, 2:7] <- kNN(petrole[, 2:7], k=5, imp_var=FALSE)
# Vérifier
head(petrole)
# Vérifier s'il reste des NA
colSums(is.na(petrole))
 
# Afficher les valeurs manquantes avant KNN
cat("Valeurs manquantes GAZ avant KNN :\n")
print(colSums(is.na(gaz)))

# Appliquer KNN (k=5 voisins) sur les colonnes numériques
gaz[, 2:7] <- kNN(gaz[, 2:7], k=5, imp_var=FALSE)

# Vérifier les premières lignes
head(gaz)

# Vérifier s'il reste des NA après KNN
cat("Valeurs manquantes GAZ après KNN :\n")
print(colSums(is.na(gaz)))

library(xts)

# Transformer petrole en série temporelle multidimensionnelle
petrole_xts <- xts(petrole[, 2:7], order.by=as.Date(petrole$date))

# Vérifier
head(petrole_xts)
str(petrole_xts)

# Même chose pour gaz
gaz_xts <- xts(gaz[, 2:7], order.by=as.Date(gaz$date))
head(gaz_xts)

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

# Dimensions
dim(petrole_train)  # lignes = 90% des données, colonnes = 6 variables
dim(petrole_test)

dim(gaz_train)
dim(gaz_test)

# Aperçu
head(petrole_train)
head(gaz_train)
is.numeric(petrole_train$`CL=F.Close`)
class(petrole_train)
names(petrole_train)
petrole_train$CL.F.Close <- as.numeric(petrole_train$CL.F.Close)

# 2. ANALYSE EXPLORATOIRE
library(ggplot2)
library(gridExtra)

# Graphique du pétrole
p1 <- ggplot(petrole, aes(x = as.Date(date), y = CL.F.Close)) +
  geom_line(color = "steelblue") +
  labs(title = "Évolution du prix du pétrole (WTI)",
       x = "Date", y = "Prix de clôture (USD)") +
  theme_minimal()

# Graphique du gaz
p2 <- ggplot(gaz, aes(x = as.Date(date), y = UNG.Close)) +
  geom_line(color = "darkgreen") +
  labs(title = "Évolution du prix du gaz naturel (UNG)",
       x = "Date", y = "Prix de clôture (USD)") +
  theme_minimal()

# Afficher côte à côte
grid.arrange(p1, p2, ncol = 1)


library(ggplot2)
library(zoo)

# Lissage par moyenne mobile sur 30 jours
petrole$MA30 <- zoo::rollmean(petrole$CL.F.Close, k = 30, fill = NA)
gaz$MA30 <- zoo::rollmean(gaz$UNG.Close, k = 30, fill = NA)

# Graphique PÉTROLE
ggplot(petrole, aes(x = as.Date(date))) +
  geom_line(aes(y = CL.F.Close), color = "steelblue", alpha = 0.6) +
  geom_line(aes(y = MA30), color = "red", size = 1) +
  labs(title = "Pétrole WTI : Tendance (moyenne mobile 30 jours)",
       x = "Date", y = "Prix de clôture (USD)") +
  theme_minimal()

# Graphique GAZ
ggplot(gaz, aes(x = as.Date(date))) +
  geom_line(aes(y = UNG.Close), color = "darkgreen", alpha = 0.6) +
  geom_line(aes(y = MA30), color = "red", size = 1) +
  labs(title = "Gaz Naturel : Tendance (moyenne mobile 30 jours)",
       x = "Date", y = "Prix de clôture (USD)") +
  theme_minimal()
#Décomposer la série
# Transformer en ts mensuelle
petrole_ts <- ts(petrole$CL.F.Close, frequency = 12)
gaz_ts <- ts(gaz$UNG.Close, frequency = 12)

# Décomposition
petrole_decomp <- decompose(petrole_ts)
gaz_decomp <- decompose(gaz_ts)

# Visualisation
plot(petrole_decomp)
plot(gaz_decomp)

library(tseries)

# Test de stationnarité sur le pétrole
adf.test(petrole$CL.F.Close)

# Test de stationnarité sur le gaz
adf.test(gaz$UNG.Close)

# Différenciation première
petrole$diff_close <- c(NA, diff(petrole$CL.F.Close))
gaz$diff_close <- c(NA, diff(gaz$UNG.Close))

# Visualiser
par(mfrow=c(2,1))
plot(petrole$diff_close, type="l", col="steelblue",
     main="Différenciation du prix du pétrole", ylab="Différence du prix")
plot(gaz$diff_close, type="l", col="darkgreen",
     main="Différenciation du prix du gaz", ylab="Différence du prix")
par(mfrow=c(1,1))






library(tseries)
adf.test(petrole$CL.F.Close, alternative = "stationary")
# Première différenciation
petrole_diff1 <- diff(petrole$CL.F.Close)

# Nouveau test ADF
adf.test(na.omit(petrole_diff1))
par(mfrow = c(2, 1))
#ACF et PACF sur la série stationnaire
acf(na.omit(petrole_diff1), main = "ACF - Pétrole différencié")
pacf(na.omit(petrole_diff1), main = "PACF - Pétrole différencié")
# Test ADF sur les prix de clôture du gaz
adf.test(gaz$UNG.Close, alternative = "stationary")

# Première différenciation
gaz_diff1 <- diff(gaz$UNG.Close)

# Nouveau test ADF
adf.test(na.omit(gaz_diff1))
par(mfrow = c(2, 1))

acf(na.omit(gaz_diff1), main = "ACF - Gaz naturel différencié")
pacf(na.omit(gaz_diff1), main = "PACF - Gaz naturel différencié")



# Nombre de colonnes à afficher (variables numériques)
vars_pet <- colnames(petrole)[2:7]  # Open, High, Low, Close, Adj Close, Volume

# Disposition graphique : 2 lignes × 3 colonnes pour 6 variables
par(mfrow = c(2,3))

for (col in vars_pet) {
  boxplot(petrole[[col]],
          main = paste("Boxplot -", col),
          col = "skyblue",
          ylab = col,
          border = "darkblue")
}

# Réinitialiser la disposition
par(mfrow = c(1,1))


# Colonnes numériques pour gaz
vars_gaz <- colnames(gaz)[2:7]

# Disposition graphique : 2 lignes × 3 colonnes
par(mfrow = c(2,3))

for (col in vars_gaz) {
  boxplot(gaz[[col]],
          main = paste("Boxplot -", col),
          col = "lightgreen",
          ylab = col,
          border = "darkgreen")
}

# Réinitialiser la disposition
par(mfrow = c(1,1))

