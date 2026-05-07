"""
Loads the trained PINN model and provides functions to run predictions.
Can be used to predict single plate configurations or large batches for the optimizer.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

_model = keras.models.load_model('pinn_plate_model.keras', compile=False)
_x_mean = np.load('x_mean.npy')
_x_std  = np.load('x_std.npy')

FEATURE_ORDER = ['plate_length','plate_breadth','tp','q_load','BC',
                 'nx','ny','dsx','wsx','dsy','wsy']


def predict_combined(plate_length, plate_breadth, tp, q_load, BC,
                     nx, ny, dsx, wsx, dsy, wsy):
    """
    Takes a single plate setup, normalizes the inputs, passes them to the 
    neural network, and returns the predicted stress and deflection.
    """
    nx = int(nx)
    ny = int(ny)
    dsx = float(dsx) if nx > 0 else 0.0
    wsx = float(wsx) if nx > 0 else 0.0
    dsy = float(dsy) if ny > 0 else 0.0
    wsy = float(wsy) if ny > 0 else 0.0

    x_raw = np.array([[
        plate_length, plate_breadth, tp, q_load, BC,
        nx, ny, dsx, wsx, dsy, wsy
    ]], dtype=np.float32)

    x_norm = (x_raw - _x_mean) / (_x_std + 1e-9)
    pred   = _model.predict(x_norm, verbose=0)[0]

    return {
        'deflection': max(float(pred[0]), 1e-6),
        'stress':     max(float(pred[1]), 1e-3),
        'nat_freq':   max(float(pred[2]), 0.1),
    }


def predict_batch(rows: list) -> list:
    """
    Same as predict_combined but runs a whole batch at once.
    This is much faster when the genetic algorithm needs to evaluate 
    hundreds of designs simultaneously.
    """
    if not rows:
        return []

    X = []
    for row in rows:
        nx, ny = int(row['nx']), int(row['ny'])
        dsx = float(row['dsx']) if nx > 0 else 0.0
        wsx = float(row['wsx']) if nx > 0 else 0.0
        dsy = float(row['dsy']) if ny > 0 else 0.0
        wsy = float(row['wsy']) if ny > 0 else 0.0

        X.append([
            row['plate_length'], row['plate_breadth'],
            row['tp'], row['q_load'], row['BC'],
            nx, ny, dsx, wsx, dsy, wsy
        ])

    X = np.array(X, dtype=np.float32)
    X_norm = (X - _x_mean) / (_x_std + 1e-9)
    preds = _model.predict(X_norm, verbose=0)

    return [
        {
            'deflection': max(float(p[0]), 1e-6),
            'stress':     max(float(p[1]), 1e-3),
            'nat_freq':   max(float(p[2]), 0.1),
        }
        for p in preds
    ]