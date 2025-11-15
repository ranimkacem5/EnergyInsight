# ----------------------------
# 1️⃣ Imports et préparation
# ----------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

# Chemin Kaggle
path = "/kaggle/input/energy-gp/"
pet  = pd.read_csv("CL_F.csv")
gas = pd.read_csv(path + "UNG.csv", parse_dates=["date"])

# Choisir dataset
df = pet.copy()  # ou df = gas.copy()
df = df.sort_values("date").reset_index(drop=True)

# Colonne prix
if "CL.F.Close" in df.columns:
    df["price"] = df["CL.F.Close"]
elif "UNG.Close" in df.columns:
    df["price"] = df["UNG.Close"]
else:
    raise ValueError(f"Colonne close non trouvée. Colonnes disponibles : {df.columns.tolist()}")

# Supprimer NaN
df = df.dropna(subset=["price"])

# Normalisation
scaler = MinMaxScaler()
df["scaled"] = scaler.fit_transform(df[["price"]])

# ----------------------------
# 2️⃣ Création des séquences
# ----------------------------
def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 30
X, y = create_sequences(df["scaled"].values, seq_len=SEQ_LEN)

# Reshape pour RNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ----------------------------
# 3️⃣ Modèle LSTM
# ----------------------------
model_lstm = Sequential([
    LSTM(32, input_shape=(SEQ_LEN, 1)),
    Dense(1)
])
model_lstm.compile(optimizer=Adam(0.0005), loss='mse')

history_lstm = model_lstm.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=16,
    verbose=1
)

# Prédiction
pred_lstm = model_lstm.predict(X_test).flatten()
pred_lstm_rescaled = scaler.inverse_transform(pred_lstm.reshape(-1,1)).flatten()
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

print("LSTM RMSE:", np.sqrt(mean_squared_error(y_test_rescaled, pred_lstm_rescaled)))
print("LSTM MAE:", mean_absolute_error(y_test_rescaled, pred_lstm_rescaled))

# ----------------------------
# 4️⃣ Modèle GRU
# ----------------------------
model_gru = Sequential([
    GRU(32, input_shape=(SEQ_LEN,1)),
    Dense(1)
])
model_gru.compile(optimizer=Adam(0.0005), loss='mse')

history_gru = model_gru.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=16,
    verbose=1
)

# Prédiction
pred_gru = model_gru.predict(X_test).flatten()
pred_gru_rescaled = scaler.inverse_transform(pred_gru.reshape(-1,1)).flatten()

print("GRU RMSE:", np.sqrt(mean_squared_error(y_test_rescaled, pred_gru_rescaled)))
print("GRU MAE:", mean_absolute_error(y_test_rescaled, pred_gru_rescaled))

# ----------------------------
# 5️⃣ Comparaison visuelle
# ----------------------------
plt.figure(figsize=(12,5))
plt.plot(df["date"].values[-len(y_test_rescaled):], y_test_rescaled, label="Actual")
plt.plot(df["date"].values[-len(pred_lstm_rescaled):], pred_lstm_rescaled, label="LSTM")
plt.plot(df["date"].values[-len(pred_gru_rescaled):], pred_gru_rescaled, label="GRU")
plt.title("Comparaison LSTM vs GRU")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()