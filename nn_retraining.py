!pip install tensorflow scikit-learn pandas numpy joblib -q

import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("✓ Ready. TF version:", tf.__version__)

#load data and scale
df = pd.read_csv('/content/data/f1_data_ml_ready.csv')

feature_columns = [col for col in df.columns if col != 'FinishPosition']
X = df[feature_columns].values
y = df['FinishPosition'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"✓ Data loaded: {len(df)} records, {len(feature_columns)} features")

#biuld model
tf.keras.backend.clear_session()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

model_nn = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
model_nn.summary()
print("✓ Model built")

#train
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)

history = model_nn.fit(
    X_train_scaled, y_train,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model_nn.predict(X_test_scaled, verbose=0).flatten()
print(f"\n✓ Test MAE : {mean_absolute_error(y_test, y_pred):.3f}")
print(f"✓ Test R²  : {r2_score(y_test, y_pred):.3f}")

#save model
import os

# Save scaler
joblib.dump(scaler, '/content/data/f1_scaler.pkl')
print("✓ Scaler saved")

# Save NN as Windows-compatible pkl
nn_weights    = model_nn.get_weights()
nn_config_json = model_nn.to_json()

joblib.dump({
    'weights':     nn_weights,
    'config_json': nn_config_json,
    'input_dim':   int(X_train_scaled.shape[1])
}, '/content/data/f1_nn_weights.pkl')
print("✓ NN weights saved")

# Download all 3 files
from google.colab import files

for f in ['/content/data/f1_nn_weights.pkl',
          '/content/data/f1_scaler.pkl']:
    files.download(f)
    print(f"✓ Downloaded: {os.path.basename(f)}")

print("\n✅ Done! Move both files into your Windsurf data/ folder")