# ===============================
# run_pipeline.py
# Pipeline pour exécuter tous les modèles et enregistrer RMSE/MAE
# ===============================
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, SimpleRNN, Input
from tensorflow.keras.callbacks import EarlyStopping
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# Fonction metrics
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# -----------------------------
# Charger dataset
# -----------------------------
df = pd.read_csv("CL_F.csv")
df = df.rename(columns={df.columns[0]: "date"})
df["date"] = pd.to_datetime(df["date"], errors="coerce")
close_col = [col for col in df.columns if "Close" in col]
df["value"] = pd.to_numeric(df[close_col[0]], errors="coerce")
df = df.dropna(subset=["date","value"])
df = df[df["value"]>0].reset_index(drop=True)

results = {}

# ==========================================================
# Prophet + RNN
# ==========================================================
def prophet_rnn(df):
    df["value_log"] = np.log1p(df["value"])
    prophet_df = df[["date","value_log"]].rename(columns={"date":"ds","value_log":"y"})
    model_prophet = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model_prophet.fit(prophet_df)
    forecast = model_prophet.predict(model_prophet.make_future_dataframe(periods=0))
    df["prophet_pred"] = forecast["yhat"].values
    df["residuals"] = df["value_log"] - df["prophet_pred"]

    scaler = StandardScaler()
    res_scaled = scaler.fit_transform(df[["residuals"]])

    SEQ_LEN = 30
    def create_sequences(data):
        X, y = [], []
        for i in range(len(data)-SEQ_LEN):
            seq = data[i:i+SEQ_LEN]
            target = data[i+SEQ_LEN]
            if not np.isnan(seq).any() and not np.isnan(target):
                X.append(seq)
                y.append(target)
        return np.array(X).reshape(-1, SEQ_LEN,1), np.array(y)

    X, y_seq = create_sequences(res_scaled)
    split = int(0.8*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model_rnn = Sequential([
        Input(shape=(SEQ_LEN,1)),
        SimpleRNN(64),
        Dropout(0.2),
        Dense(1)
    ])
    model_rnn.compile(optimizer='adam', loss='mse')
    model_rnn.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    
    rnn_pred = model_rnn.predict(X_test)
    rnn_pred_rescaled = scaler.inverse_transform(rnn_pred).reshape(-1)
    prophet_test = df["prophet_pred"].values[-len(rnn_pred):]
    hybrid_pred = np.expm1(prophet_test + rnn_pred_rescaled)
    y_true = np.expm1(df["value_log"].values[-len(hybrid_pred):])
    return {"RMSE": rmse(y_true, hybrid_pred), "MAE": mae(y_true, hybrid_pred)}

# ==========================================================
# LSTM
# ==========================================================
def lstm_model(df):
    scaler = MinMaxScaler()
    df["scaled"] = scaler.fit_transform(df[["value"]])
    SEQ_LEN = 30
    def create_sequences(data):
        X, y = [], []
        for i in range(len(data)-SEQ_LEN):
            X.append(data[i:i+SEQ_LEN])
            y.append(data[i+SEQ_LEN])
        return np.array(X), np.array(y)

    X, y = create_sequences(df["scaled"].values)
    X = X.reshape((X.shape[0], X.shape[1],1))
    split = int(0.8*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(32, input_shape=(SEQ_LEN,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    pred = model.predict(X_test).flatten()
    pred_rescaled = scaler.inverse_transform(pred.reshape(-1,1)).flatten()
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    return {"RMSE": rmse(y_test_rescaled, pred_rescaled), "MAE": mae(y_test_rescaled, pred_rescaled)}

# ==========================================================
# GRU
# ==========================================================
def gru_model(df):
    scaler = MinMaxScaler()
    df["scaled"] = scaler.fit_transform(df[["value"]])
    SEQ_LEN = 30
    def create_sequences(data):
        X, y = [], []
        for i in range(len(data)-SEQ_LEN):
            X.append(data[i:i+SEQ_LEN])
            y.append(data[i+SEQ_LEN])
        return np.array(X), np.array(y)

    X, y = create_sequences(df["scaled"].values)
    X = X.reshape((X.shape[0], X.shape[1],1))
    split = int(0.8*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        GRU(32, input_shape=(SEQ_LEN,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    pred = model.predict(X_test).flatten()
    pred_rescaled = scaler.inverse_transform(pred.reshape(-1,1)).flatten()
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    return {"RMSE": rmse(y_test_rescaled, pred_rescaled), "MAE": mae(y_test_rescaled, pred_rescaled)}

# ==========================================================
# ARIMA + LSTM
# ==========================================================
def arima_lstm(df):
    # Transformation log pour stabiliser la variance
    df["value_log"] = np.log1p(df["value"])
    
    # ARIMA
    model_arima = ARIMA(df["value_log"], order=(2,1,2)).fit()
    df["arima_pred"] = model_arima.predict(start=0, end=len(df)-1)
    
    # Résidus ARIMA
    df["residuals"] = df["value_log"] - df["arima_pred"]
    
    # Standardisation des résidus
    scaler = StandardScaler()
    res_scaled = scaler.fit_transform(df[["residuals"]])
    
    SEQ_LEN = 30
    def create_sequences(data):
        X, y = [], []
        for i in range(len(data)-SEQ_LEN):
            seq = data[i:i+SEQ_LEN]
            target = data[i+SEQ_LEN]
            if not np.isnan(seq).any() and not np.isnan(target):
                X.append(seq)
                y.append(target)
        return np.array(X).reshape(-1, SEQ_LEN, 1), np.array(y)
    
    X, y = create_sequences(res_scaled)
    split = int(0.8*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # LSTM sur résidus
    model = Sequential([
        Input(shape=(SEQ_LEN,1)),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    
    # Prédiction LSTM
    pred_resid = model.predict(X_test).reshape(-1,1)
    pred_resid_inv = scaler.inverse_transform(pred_resid).reshape(-1)
    
    # Prédiction ARIMA correspondante pour le test set
    arima_test = df["arima_pred"].values[-len(pred_resid_inv):]
    
    # Prédiction finale hybride (log -> valeur originale)
    hybrid_pred = np.expm1(arima_test + pred_resid_inv)
    y_true = np.expm1(df["value_log"].values[-len(hybrid_pred):])
    
    return {"RMSE": rmse(y_true, hybrid_pred), "MAE": mae(y_true, hybrid_pred)}

# ==========================================================
# Exécuter pipeline
# ==========================================================
print("🚀 Running Prophet+RNN...")
results["Prophet+RNN"] = prophet_rnn(df)

print("🚀 Running LSTM...")
results["LSTM"] = lstm_model(df)

print("🚀 Running GRU...")
results["GRU"] = gru_model(df)

print("🚀 Running ARIMA+LSTM...")
results["ARIMA+LSTM"] = arima_lstm(df)

# Sauvegarder dans JSON
with open("data/results_metricspython.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Pipeline terminé ranim. Résultats sauvegardés dans results_metrics.json")
