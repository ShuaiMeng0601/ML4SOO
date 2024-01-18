def normalize(data):
    import numpy as np
    mean = np.mean(data)
    normalized_data = (data - mean) / (np.std(data))

    return normalized_data

def L2_loss(y_true, y_pred):
    import tensorflow as tf
    from tensorflow import keras
    return keras.backend.mean(keras.backend.square(y_true-y_pred),axis=-1)