"""Optional check: compare the app's NumPy inference with Keras itself.

Run locally (needs TensorFlow, which the deployed app does NOT need):
    pip install tensorflow keras==3.13.2
    python scripts/verify_with_keras.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import keras  # noqa: E402

from recommender import MODEL_PATH, Recommender  # noqa: E402

rec = Recommender()
model = keras.models.load_model(MODEL_PATH)
rng = np.random.default_rng(0)
X = np.column_stack([rng.integers(0, rec.n_users, 5000),
                     rng.integers(0, rec.n_movies, 5000) + rec.n_users])
keras_pred = model.predict(X, verbose=0).flatten()
numpy_pred = rec.model.predict_raw(X)
diff = np.abs(keras_pred - numpy_pred).max()
print(f"max |keras - numpy| over 5000 pairs = {diff:.2e}")
assert diff < 1e-4, "Mismatch between Keras and the app's inference!"
print("OK - the app reproduces the Keras model's predictions.")
