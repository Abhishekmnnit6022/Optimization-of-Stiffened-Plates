"""
Script to train the Neural Network.
It loads the FEA dataset, augments the loads, normalizes the inputs, 
and trains a simple Keras model to predict structural responses.
Saves the model and scaling parameters when done.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from fea_data import BASE_L, BASE_B, BASE_Q, RAW_FEA

cols = ['BC','tp','dsx','wsx','dsy','wsy','nx','ny',
        'fea_deflection','fea_stress','fea_nat_freq']
df_base = pd.DataFrame(RAW_FEA, columns=cols)
df_base['plate_length']  = BASE_L
df_base['plate_breadth'] = BASE_B
df_base['q_load']        = BASE_Q
# Multiply the base dataset by different load factors (q) to generate more training data.
# Deflection and stress scale linearly with load, so we just multiply them.
Q_AUGMENT = [0.5, 1.0, 2.0, 5.0]
rows = []
for q in Q_AUGMENT:
    df_q = df_base.copy()
    df_q['q_load']         = q
    df_q['fea_deflection'] = df_base['fea_deflection'] * q
    df_q['fea_stress']     = df_base['fea_stress']     * q
    df_q['fea_nat_freq']   = df_base['fea_nat_freq']
    rows.append(df_q)
df = pd.concat(rows, ignore_index=True)

# Replace missing frequencies with mean of valid ones
valid_freq = df.loc[df['fea_nat_freq'] > 0, 'fea_nat_freq']
df.loc[df['fea_nat_freq'] <= 0, 'fea_nat_freq'] = valid_freq.mean()

FEATURE_COLS = ['plate_length','plate_breadth','tp','q_load','BC',
                'nx','ny','dsx','wsx','dsy','wsy']
TARGET_COLS  = ['fea_deflection','fea_stress','fea_nat_freq']

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COLS].values.astype(np.float32)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

np.save('x_mean.npy', scaler_X.mean_.astype(np.float32))
np.save('x_std.npy',  np.sqrt(scaler_X.var_).astype(np.float32))

model = keras.Sequential([
    keras.layers.Input(shape=(11,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64,  activation='relu'),
    keras.layers.Dense(32,  activation='relu'),
    keras.layers.Dense(3)   # [deflection, stress, nat_freq]
])

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', patience=300,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=150, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint('pinn_plate_model.keras',
                                    monitor='loss', save_best_only=True, verbose=1)
]

print(f"Training on {len(X)} samples...")
history = model.fit(X_scaled, y, epochs=3000, batch_size=32,
                    verbose=2, callbacks=callbacks)

print("\nTraining complete. Saved files:")
print("  pinn_plate_model.keras")
print("  x_mean.npy")
print("  x_std.npy")