!pip install tensorflow scikit-learn pandas numpy joblib -q

import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Check GPU
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)

# Load processed ML-ready data
df = pd.read_csv('/content/data/f1_data_ml_ready.csv')

print(f"✓ Loaded {len(df)} records")
print(f"✓ Features: {len(df.columns) - 1}")

# Separate features and target
feature_columns = [col for col in df.columns if col != 'FinishPosition']
X = df[feature_columns].values
y = df['FinishPosition'].values

# Split FIRST, then scale (important - prevents data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (neural networks need this, tree models don't)
# StandardScaler: transforms each feature to mean=0, std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit AND transform on training
X_test_scaled = scaler.transform(X_test)         # only transform on test (no fit!)

print(f"\n✓ Training set: {len(X_train)} records")
print(f"✓ Test set: {len(X_test)} records")
print(f"✓ Features scaled: mean~0, std~1")
print(f"\nExample - GridPosition before scaling: {X_train[:3, 0]}")
print(f"Example - GridPosition after scaling:  {X_train_scaled[:3, 0]}")

#Biulding the NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Clear any existing model from memory first
import tensorflow as tf
tf.keras.backend.clear_session()

def build_f1_neural_network(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),          # Use Input layer instead of input_shape=
        
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
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    
    return model

model_nn = build_f1_neural_network(input_dim=X_train_scaled.shape[1])
model_nn.summary()

print("\n✓ Neural network built!")
print(f"  Input neurons: {X_train_scaled.shape[1]}")
print(f"  Output neurons: 1 (predicted position)")

#training the network
# Callbacks - these automatically improve training

# Stop training if validation loss stops improving
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,           # Wait 20 epochs before stopping
    restore_best_weights=True,  # Keep best weights, not last
    verbose=1
)

# Reduce learning rate if stuck
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,            # Halve the learning rate
    patience=10,
    min_lr=0.00001,
    verbose=1
)

print("="*70)
print("TRAINING F1 NEURAL NETWORK")
print("="*70)
print("Watch the val_mae - this is what matters!")
print("Lower = better predictions\n")

history = model_nn.fit(
    X_train_scaled, y_train,
    validation_split=0.15,    # 15% of training used for validation
    epochs=200,               # Max epochs (early stopping will likely kick in)
    batch_size=32,            # Process 32 records at a time
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Final evaluation
print("\n" + "="*70)
print("NEURAL NETWORK RESULTS")
print("="*70)

y_pred_train = model_nn.predict(X_train_scaled, verbose=0).flatten()
y_pred_test = model_nn.predict(X_test_scaled, verbose=0).flatten()

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n  Train MAE : {train_mae:.3f} positions")
print(f"  Test MAE  : {test_mae:.3f} positions")
print(f"  Test R²   : {test_r2:.3f}")

if test_mae < 3.0:
    print(f"\n  ✅ IMPROVEMENT over Gradient Boosting (3.209)!")
elif test_mae < 3.209:
    print(f"\n  ✅ Slightly better than Gradient Boosting!")
else:
    print(f"\n  ⚠️  Similar to Gradient Boosting — ensemble will help")

#plot training progress
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#1a1a2e')

for ax in axes:
    ax.set_facecolor('#0f3460')

# Plot 1: Loss over epochs
axes[0].plot(history.history['loss'], color='#e94560', label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], color='#00d4ff', label='Validation Loss', linewidth=2)
axes[0].set_title('Loss During Training', color='white', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', color='white')
axes[0].set_ylabel('Huber Loss', color='white')
axes[0].legend(facecolor='#1a1a2e', labelcolor='white')
axes[0].tick_params(colors='white')

# Plot 2: MAE over epochs
axes[1].plot(history.history['mae'], color='#e94560', label='Training MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], color='#00d4ff', label='Validation MAE', linewidth=2)
axes[1].axhline(y=3.209, color='#FFD700', linestyle='--', linewidth=2, label='GBM baseline (3.209)')
axes[1].set_title('MAE During Training', color='white', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', color='white')
axes[1].set_ylabel('Mean Absolute Error', color='white')
axes[1].legend(facecolor='#1a1a2e', labelcolor='white')
axes[1].tick_params(colors='white')

plt.tight_layout()
plt.savefig('/content/data/nn_training_history.png', dpi=300, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.show()

print(f"\nEpochs trained: {len(history.history['loss'])}")
print(f"Best val_mae: {min(history.history['val_mae']):.3f}")

#biuld ensemlble(nn + gradien boosting)
# Load the existing Gradient Boosting model
gbm_model = joblib.load('/content/data/f1_best_model.pkl')

# Ensemble: average predictions from both models
# This usually beats either model alone

def ensemble_predict(X_raw, X_scaled):
    """Combine GBM and NN predictions."""
    gbm_pred = gbm_model.predict(X_raw)
    nn_pred = model_nn.predict(X_scaled, verbose=0).flatten()
    
    # Weighted average: give slightly more weight to whichever performed better
    gbm_weight = 0.5
    nn_weight = 0.5
    
    return (gbm_pred * gbm_weight) + (nn_pred * nn_weight)

# Use raw (unscaled) for GBM, scaled for NN
X_test_df = pd.DataFrame(X_test, columns=feature_columns)

ensemble_pred = ensemble_predict(X_test_df, X_test_scaled)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)

print("="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"\n  Gradient Boosting : MAE = 3.209 | R² = 0.425")
print(f"  Neural Network    : MAE = {test_mae:.3f} | R² = {test_r2:.3f}")
print(f"  Ensemble (50/50)  : MAE = {ensemble_mae:.3f} | R² = {ensemble_r2:.3f}")

# Find best model
models = {
    'Gradient Boosting': 3.209,
    'Neural Network': test_mae,
    'Ensemble': ensemble_mae
}
best = min(models, key=models.get)
print(f"\n  🏆 WINNER: {best} (MAE = {models[best]:.3f})")

