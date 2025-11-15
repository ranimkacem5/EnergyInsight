# ===============================
#  1. Importation des librairies
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===============================
#  2. Chargement du dataset
# ===============================
data = pd.read_csv("CL_F.csv")
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

features = ['CL.F.Open', 'CL.F.High', 'CL.F.Low', 'CL.F.Close', 'CL.F.Volume', 'CL.F.Adjusted']
df = data[features].copy()

# ===============================
#  3. Feature engineering
# ===============================

# Log du volume (plus stable)
df["CL.F.Volume"] = np.log1p(df["CL.F.Volume"])

# Indicateurs techniques
df["MA7"] = df["CL.F.Close"].rolling(7).mean()
df["MA21"] = df["CL.F.Close"].rolling(21).mean()
df["Volatility"] = df["CL.F.Close"].pct_change().rolling(7).std()

# Returns
df["Returns"] = df["CL.F.Close"].pct_change()

df = df.dropna()

# ===============================
#  4. Scaling
# ===============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# ===============================
#  5. Fonction pour créer les séquences
# ===============================
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, df.columns.get_loc("CL.F.Close")])
    return np.array(X), np.array(y)

SEQ_LEN = 60
X, y = create_sequences(scaled_data, SEQ_LEN)

train_size = int(0.9 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ===============================
#  6. Modèle LSTM amélioré
# ===============================

model_lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(32, activation="relu"),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mse')

# ===============================
#  7. Modèle GRU amélioré
# ===============================

model_gru = Sequential([
    GRU(128, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    GRU(64),
    Dense(32, activation="relu"),
    Dense(1)
])

model_gru.compile(optimizer='adam', loss='mse')

# ===============================
#  8. Callbacks
# ===============================
early = EarlyStopping(patience=5, restore_best_weights=True)
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# ===============================
#  9. Entraînement LSTM & GRU
# ===============================

history_lstm = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    callbacks=[early, reduceLR],
    verbose=1
)

history_gru = model_gru.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    callbacks=[early, reduceLR],
    verbose=1
)

# ===============================
#  10. Prédictions
# ===============================

pred_scaled_lstm = model_lstm.predict(X_test)
pred_scaled_gru = model_gru.predict(X_test)

# Reconstruction inverse
def inverse_close(pred_scaled, y_test):
    empty_pred = np.zeros((len(pred_scaled), df.shape[1]))
    empty_pred[:, df.columns.get_loc("CL.F.Close")] = pred_scaled[:, 0]

    empty_y = np.zeros((len(y_test), df.shape[1]))
    empty_y[:, df.columns.get_loc("CL.F.Close")] = y_test

    inv_pred = scaler.inverse_transform(empty_pred)[:, df.columns.get_loc("CL.F.Close")]
    inv_y = scaler.inverse_transform(empty_y)[:, df.columns.get_loc("CL.F.Close")]

    return inv_pred, inv_y

inv_pred_lstm, inv_y = inverse_close(pred_scaled_lstm, y_test)
inv_pred_gru, inv_y2 = inverse_close(pred_scaled_gru, y_test)

# ===============================
#  11. Graphiques comparatifs
# ===============================

plt.figure(figsize=(12,5))
plt.plot(inv_y, label="Prix réel")
plt.plot(inv_pred_lstm, label="LSTM prédiction")
plt.title("Comparaison LSTM vs Réel")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(inv_y2, label="Prix réel")
plt.plot(inv_pred_gru, label="GRU prédiction")
plt.title("Comparaison GRU vs Réel")
plt.legend()
plt.show()

# ===============================
#  12. Comparaison directe LSTM vs GRU
# ===============================
plt.figure(figsize=(12,5))
plt.plot(inv_y, label="Vrai")
plt.plot(inv_pred_lstm, label="LSTM")
plt.plot(inv_pred_gru, label="GRU")
plt.title("LSTM vs GRU vs Réel")
plt.legend()
plt.show()
