import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
CSV_PATH = r"D:\rPPG_Results\rppg_features.csv"
FEATURES = ['heart_rate', 'hrv_sdnn', 'hrv_rmssd', 'signal_mean', 'signal_std', 'signal_min', 'signal_max']
RANDOM_STATE = 42

# Load data
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} samples")
print(f"Drunk: {len(df[df['category']=='drunk'])}, Sober: {len(df[df['category']=='sober'])}\n")

# Prepare data
X = df[FEATURES].values
y = (df['category'] == 'drunk').astype(int).values

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM: (samples, timesteps, features)
X_train_lstm = np.expand_dims(X_train_scaled, axis=2)
X_test_lstm = np.expand_dims(X_test_scaled, axis=2)

print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"LSTM input shape: {X_train_lstm.shape}\n")

# Build LSTM Model
print("Building LSTM Model...")
model = Sequential([
    LSTM(64, activation='relu', input_shape=(7, 1), return_sequences=True),
    Dropout(0.2),
    
    LSTM(32, activation='relu', return_sequences=False),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

print("Training LSTM...\n")
history = model.fit(
    X_train_lstm, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

print(f"Training complete (stopped at epoch {len(history.history['loss'])})\n")

# Evaluate
y_pred_lstm = (model.predict(X_test_lstm, verbose=0) > 0.5).astype(int).flatten()

lstm_accuracy = accuracy_score(y_test, y_pred_lstm)
lstm_precision = precision_score(y_test, y_pred_lstm, zero_division=0)
lstm_recall = recall_score(y_test, y_pred_lstm, zero_division=0)
lstm_f1 = f1_score(y_test, y_pred_lstm, zero_division=0)

print(f"Accuracy:  {lstm_accuracy:.4f}")
print(f"Precision: {lstm_precision:.4f}")
print(f"Recall:    {lstm_recall:.4f}")
print(f"F1-Score:  {lstm_f1:.4f}\n")
