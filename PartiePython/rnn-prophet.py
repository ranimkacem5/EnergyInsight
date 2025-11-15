# ----------------------------
# 1️⃣ Imports
# ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from prophet import Prophet

# ----------------------------
# 2️⃣ Charger dataset
# ----------------------------
df = pd.read_csv("CL_F.csv")
df = df.rename(columns={df.columns[0]: "date"})
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Détecter colonne Close automatiquement
close_col = [col for col in df.columns if "Close" in col]
if not close_col:
    raise ValueError("Aucune colonne Close trouvée !")
df["value"] = pd.to_numeric(df[close_col[0]], errors="coerce")

# Nettoyage
df = df.dropna(subset=["date", "value"])
df = df[df["value"] > 0].reset_index(drop=True)  # >0 pour log-transform
df = df[df["value"].replace([np.inf, -np.inf], np.nan).notna()]

# Transformation log pour stabiliser variance
df["value_log"] = np.log1p(df["value"])

# ----------------------------
# 3️⃣ Prophet pour tendance
# ----------------------------
prophet_df = df[["date", "value_log"]].rename(columns={"date":"ds", "value_log":"y"})
model_prophet = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
model_prophet.fit(prophet_df)

future = model_prophet.make_future_dataframe(periods=0)  # même période que df
forecast = model_prophet.predict(future)
df["prophet_pred"] = forecast["yhat"].values

# Résidus pour RNN
df["residuals"] = df["value_log"] - df["prophet_pred"]

# ----------------------------
# 4️⃣ Normalisation des résidus
# ----------------------------
scaler = StandardScaler()
res_scaled = scaler.fit_transform(df[["residuals"]])

# ----------------------------
# 5️⃣ Création des séquences RNN
# ----------------------------
SEQ_LEN = 30
def create_sequences(data, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(data)-seq_len):
        seq = data[i:i+seq_len]
        target = data[i+seq_len]
        if not np.isnan(seq).any() and not np.isnan(target):
            X.append(seq)
            y.append(target)
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)
    return X, y

X, y = create_sequences(res_scaled, SEQ_LEN)

# Split train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ----------------------------
# 6️⃣ RNN amélioré
# ----------------------------
print("🔵 Training RNN...")
model_rnn = Sequential([
    Input(shape=(SEQ_LEN,1)),
    SimpleRNN(64, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model_rnn.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model_rnn.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ----------------------------
# 7️⃣ Prédiction hybride
# ----------------------------
rnn_pred = model_rnn.predict(X_test)
rnn_pred_rescaled = scaler.inverse_transform(rnn_pred)

# Prophet correspondante à la période de test
prophet_test = df["prophet_pred"].values[-len(rnn_pred):]

# Hybrid = Prophet + RNN
hybrid_pred_log = prophet_test + rnn_pred_rescaled.reshape(-1)

# Retour à l'échelle originale
hybrid_pred = np.expm1(hybrid_pred_log)
y_test_orig = np.expm1(df["value_log"].values[-len(hybrid_pred):])

# ----------------------------
# 8️⃣ Évaluation
# ----------------------------
rmse = np.sqrt(mean_squared_error(y_test_orig, hybrid_pred))
mae = mean_absolute_error(y_test_orig, hybrid_pred)
print(f"Hybrid Prophet+RNN RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# ----------------------------
# 9️⃣ Visualisation
# ----------------------------
plt.figure(figsize=(12,5))
plt.plot(df["date"], df["value"], label="Actual")
plt.plot(df["date"].values[-len(hybrid_pred):], hybrid_pred, label="Hybrid Prophet+RNN")
plt.title("Prévision Hybrid Prophet + RNN")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.legend()
plt.show()

# ----------------------------
# 10️⃣ Fonction prédiction vs réel
# ----------------------------
def plot_pred_vs_real(y_true, y_pred, title="Prédictions vs Réel"):
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label="Réel", color='blue')
    plt.plot(y_pred, label="Prédit", color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Valeur")
    plt.legend()
    plt.show()

# Appel de la fonction
plot_pred_vs_real(y_test_orig, hybrid_pred, title="Hybrid Prophet+RNN: Prédiction vs Réel")
