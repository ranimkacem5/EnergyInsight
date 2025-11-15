# ==========================================================
# 03_modeles_classiques.R (amélioré)
# ==========================================================

library(forecast)
library(tseries)
library(vars)
library(rugarch)
library(Metrics)
library(jsonlite)

load("data/splits.RData")

# --- Extraction des sous-séries ---
petrole_train_df <- data.frame(date = index(petrole_split$train), CL.F.Close = petrole_split$train$CL.F.Close)
petrole_test_df  <- data.frame(date = index(petrole_split$test), CL.F.Close = petrole_split$test$CL.F.Close)
gaz_train_df <- data.frame(date = index(gaz_split$train), UNG.Close = gaz_split$train$UNG.Close)
gaz_test_df  <- data.frame(date = index(gaz_split$test), UNG.Close = gaz_split$test$UNG.Close)

# ==========================================================
# Fonction utilitaire pour MAPE
# ==========================================================
mape_safe <- function(actual, predicted) {
  mean(abs((actual - predicted)/actual)) * 100
}

# ==========================================================
# ARIMA
# ==========================================================
auto_arima_pet <- auto.arima(petrole_train_df$CL.F.Close, seasonal = FALSE)
auto_arima_gaz <- auto.arima(gaz_train_df$UNG.Close, seasonal = FALSE)

forecast_pet <- forecast(auto_arima_pet, h = nrow(petrole_test_df))
forecast_gaz <- forecast(auto_arima_gaz, h = nrow(gaz_test_df))

# Metrics ARIMA
rmse_pet <- rmse(petrole_test_df$CL.F.Close, as.numeric(forecast_pet$mean))
mae_pet  <- mae(petrole_test_df$CL.F.Close, as.numeric(forecast_pet$mean))
mape_pet <- mape_safe(petrole_test_df$CL.F.Close, as.numeric(forecast_pet$mean))

rmse_gaz <- rmse(gaz_test_df$UNG.Close, as.numeric(forecast_gaz$mean))
mae_gaz  <- mae(gaz_test_df$UNG.Close, as.numeric(forecast_gaz$mean))
mape_gaz <- mape_safe(gaz_test_df$UNG.Close, as.numeric(forecast_gaz$mean))

# ==========================================================
# SARIMA
# ==========================================================
sarima_pet <- auto.arima(petrole_train_df$CL.F.Close, seasonal = TRUE)
sarima_gaz <- auto.arima(gaz_train_df$UNG.Close, seasonal = TRUE)

forecast_pet_sarima <- forecast(sarima_pet, h = nrow(petrole_test_df))
forecast_gaz_sarima <- forecast(sarima_gaz, h = nrow(gaz_test_df))

rmse_pet_sarima <- rmse(petrole_test_df$CL.F.Close, forecast_pet_sarima$mean)
mae_pet_sarima  <- mae(petrole_test_df$CL.F.Close, forecast_pet_sarima$mean)
mape_pet_sarima <- mape_safe(petrole_test_df$CL.F.Close, forecast_pet_sarima$mean)

# ==========================================================
# VAR (alignement automatique)
# ==========================================================
common_dates <- intersect(index(petrole_split$train), index(gaz_split$train))
petrole_train_aligned <- petrole_split$train[common_dates, "CL.F.Close"]
gaz_train_aligned     <- gaz_split$train[common_dates, "UNG.Close"]

var_train <- na.omit(data.frame(
  petrole = diff(petrole_train_aligned),
  gaz = diff(gaz_train_aligned)
))

lag_sel <- VARselect(var_train, lag.max = 10)$selection["AIC(n)"]
var_model <- VAR(var_train, p = lag_sel)
var_forecast <- predict(var_model, n.ahead = nrow(petrole_test_df))

forecast_pet_var <- var_forecast$fcst[["petrole"]][, "fcst"]
forecast_gaz_var <- var_forecast$fcst[["gaz"]][, "fcst"]

var_rmse_pet <- rmse(petrole_test_df$CL.F.Close[1:length(forecast_pet_var)], forecast_pet_var)
var_mae_pet  <- mae(petrole_test_df$CL.F.Close[1:length(forecast_pet_var)], forecast_pet_var)
var_mape_pet <- mape_safe(petrole_test_df$CL.F.Close[1:length(forecast_pet_var)], forecast_pet_var)

var_rmse_gaz <- rmse(gaz_test_df$UNG.Close[1:length(forecast_gaz_var)], forecast_gaz_var)
var_mae_gaz  <- mae(gaz_test_df$UNG.Close[1:length(forecast_gaz_var)], forecast_gaz_var)
var_mape_gaz <- mape_safe(gaz_test_df$UNG.Close[1:length(forecast_gaz_var)], forecast_gaz_var)

VAR_results <- list(
  order = lag_sel,
  forecast = list(
    petrole = as.numeric(forecast_pet_var),
    gaz = as.numeric(forecast_gaz_var)
  ),
  rmse = list(petrole = var_rmse_pet, gaz = var_rmse_gaz),
  mae  = list(petrole = var_mae_pet,  gaz = var_mae_gaz),
  mape = list(petrole = var_mape_pet, gaz = var_mape_gaz)
)

# ==========================================================
# GARCH / ARCH
# ==========================================================
log_ret_pet <- na.omit(diff(log(petrole_train_df$CL.F.Close)))
spec_pet <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                       mean.model = list(armaOrder = c(1, 1)))
garch_pet <- ugarchfit(spec_pet, data = log_ret_pet)

spec_gaz <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 0)),
                       mean.model = list(armaOrder = c(1, 1)))
garch_gaz_arch <- ugarchfit(spec_gaz, data = na.omit(diff(log(gaz_train_df$UNG.Close))))

# Metrics GARCH/ARCH
garch_pet_ll <- likelihood(garch_pet)
garch_pet_aic <- infocriteria(garch_pet)["Akaike",]
garch_pet_bic <- infocriteria(garch_pet)["Bayes",]

garch_gaz_ll <- likelihood(garch_gaz_arch)
garch_gaz_aic <- infocriteria(garch_gaz_arch)["Akaike",]
garch_gaz_bic <- infocriteria(garch_gaz_arch)["Bayes",]

# ==========================================================
# Export des résultats
# ==========================================================
results <- list(
  ARIMA = list(
    petrole = list(
      order = paste0("arima(", paste(auto_arima_pet$arma[c(1,6,2)], collapse = ","), ")"),
      forecast = as.numeric(forecast_pet$mean),
      rmse = rmse_pet, mae = mae_pet, mape = mape_pet
    ),
    gaz = list(
      order = paste0("arima(", paste(auto_arima_gaz$arma[c(1,6,2)], collapse = ","), ")"),
      forecast = as.numeric(forecast_gaz$mean),
      rmse = rmse_gaz, mae = mae_gaz, mape = mape_gaz
    )
  ),
  SARIMA = list(
    petrole = list(
      order = paste0("sarima(", paste(sarima_pet$arma[c(1,6,2)], collapse = ","), ")"),
      seasonal_order = paste0("(", paste(sarima_pet$arma[c(3,7,4,5)], collapse = ","), ")"),
      forecast = as.numeric(forecast_pet_sarima$mean),
      rmse = rmse_pet_sarima, mae = mae_pet_sarima, mape = mape_pet_sarima
    ),
    gaz = list(
      order = paste0("sarima(", paste(sarima_gaz$arma[c(1,6,2)], collapse = ","), ")"),
      seasonal_order = paste0("(", paste(sarima_gaz$arma[c(3,7,4,5)], collapse = ","), ")"),
      forecast = as.numeric(forecast_gaz_sarima$mean)
    )
  ),
  VAR = VAR_results,
  GARCH = list(
    petrole = list(
      spec = list(variance.model = garch_pet@model$modeldesc$variance.model,
                  mean.model = garch_pet@model$modeldesc$mean.model),
      coef = as.list(coef(garch_pet)),
      logLik = garch_pet_ll,
      AIC = garch_pet_aic,
      BIC = garch_pet_bic
    ),
    gaz_ARCH = list(
      spec = list(variance.model = garch_gaz_arch@model$modeldesc$variance.model,
                  mean.model = garch_gaz_arch@model$modeldesc$mean.model),
      coef = as.list(coef(garch_gaz_arch)),
      logLik = garch_gaz_ll,
      AIC = garch_gaz_aic,
      BIC = garch_gaz_bic
    )
  )
)

write_json(results, path = "reports/model_results.json", pretty = TRUE, auto_unbox = TRUE)
cat("✅ Résultats et métriques exportés dans model_results.json\n")

