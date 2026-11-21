"""
Inference engine for the MovieLens Deep Learning Recommendation Scoring Model.

Everything here mirrors the logic of `MovieLens_DL2_project.ipynb`:

* Encoding  - LabelEncoder on userId / movieId. LabelEncoder sorts the unique ids, so
              the encoded index is the position of the id in `user_ids.npy` /
              `movie_ids.npy` (the saved `encoder.classes_`).
* Features  - X = [user_encoded, movie_encoded + n_users]   (get_features in the notebook)
* Model     - Embedding(n_users + n_movies, 32) -> Flatten -> Dense(128, relu)
              -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, linear)
* Output    - predicted rating, clipped to [0.5, 5.0]       (evaluate() in the notebook)

The trained weights are read directly from `final_model.keras` (no retraining).
The forward pass is computed with NumPy, using the exact same operations Keras runs
at inference time, so the app does not need TensorFlow (it is very large for
Streamlit Community Cloud). `scripts/verify_with_keras.py` checks that the
NumPy results match `keras.models.load_model(...).predict(...)`.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_model.keras"
USER_IDS_PATH = ROOT / "models" / "user_ids.npy"
MOVIE_IDS_PATH = ROOT / "models" / "movie_ids.npy"
RATINGS_PATH = ROOT / "data" / "ratings.csv"
MOVIES_PATH = ROOT / "data" / "movies.csv"
POPULARITY_PATH = ROOT / "data" / "movie_popularity.csv"

RATING_MIN, RATING_MAX = 0.5, 5.0  # same clipping as the notebook

_SUPPORTED_ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "linear": lambda x: x,
}


# --------------------------------------------------------------------------- #
# Weight reading
# --------------------------------------------------------------------------- #
WeightReader = Callable[[bytes, list[str]], dict[str, np.ndarray]]


def h5py_reader(raw: bytes, paths: list[str]) -> dict[str, np.ndarray]:
    """Read datasets from the in-memory `model.weights.h5` using h5py."""
    import h5py

    with h5py.File(io.BytesIO(raw), "r") as f:
        return {p: np.asarray(f[p][()], dtype=np.float32) for p in paths}


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


@dataclass
class _DenseLayer:
    kernel: np.ndarray
    bias: np.ndarray | None
    activation: Callable[[np.ndarray], np.ndarray]


class RatingModel:
    """The trained Keras model, loaded from a `.keras` file, run with NumPy."""

    def __init__(self, keras_path: Path | str = MODEL_PATH, reader: WeightReader = h5py_reader):
        with zipfile.ZipFile(keras_path) as z:
            config = json.loads(z.read("config.json"))
            weights_raw = z.read("model.weights.h5")

        layers = config["config"]["layers"]
        # Keras 3 stores each layer's variables under layers/<snake_class_name>[_<n>]/vars/<i>
        counters: dict[str, int] = {}
        plan: list[tuple[str, dict, str]] = []
        for layer in layers:
            cls = layer["class_name"]
            if cls == "InputLayer":
                continue
            base = _snake(cls)
            n = counters.get(base, 0)
            counters[base] = n + 1
            plan.append((cls, layer["config"], base if n == 0 else f"{base}_{n}"))

        paths: list[str] = []
        for cls, cfg, key in plan:
            if cls == "Embedding":
                paths.append(f"layers/{key}/vars/0")
            elif cls == "Dense":
                paths.append(f"layers/{key}/vars/0")
                if cfg.get("use_bias", True):
                    paths.append(f"layers/{key}/vars/1")
            elif cls != "Flatten":
                raise ValueError(f"Unsupported layer in saved model: {cls}")
        w = reader(weights_raw, paths)

        self.embedding: np.ndarray | None = None
        self.dense: list[_DenseLayer] = []
        self.summary: list[dict] = []
        for cls, cfg, key in plan:
            if cls == "Embedding":
                # float64 math: the trained float32 weights contain subnormal values that make
                # float32 matrix products ~50x slower on CPUs; results differ by < 1e-6.
                self.embedding = w[f"layers/{key}/vars/0"].astype(np.float64)
                self.summary.append({"Layer": "Embedding", "Output": f"(None, 2, {cfg['output_dim']})",
                                     "Params": int(self.embedding.size)})
            elif cls == "Flatten":
                self.summary.append({"Layer": "Flatten", "Output": "(None, 64)", "Params": 0})
            elif cls == "Dense":
                act = cfg.get("activation", "linear")
                if act not in _SUPPORTED_ACTIVATIONS:
                    raise ValueError(f"Unsupported activation: {act}")
                kernel = w[f"layers/{key}/vars/0"].astype(np.float64)
                bias = w.get(f"layers/{key}/vars/1")
                bias = None if bias is None else bias.astype(np.float64)
                self.dense.append(_DenseLayer(kernel, bias, _SUPPORTED_ACTIVATIONS[act]))
                self.summary.append({"Layer": f"Dense ({act})", "Output": f"(None, {cfg['units']})",
                                     "Params": int(kernel.size + (0 if bias is None else bias.size))})
        if self.embedding is None:
            raise ValueError("Saved model has no Embedding layer")
        self.input_dim = self.embedding.shape[0]

    def predict_raw(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Exact equivalent of `model.predict(X).flatten()` (before clipping)."""
        X = np.asarray(X, dtype=np.int64)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n, 2)")
        if X.size and (X.min() < 0 or X.max() >= self.input_dim):
            raise ValueError("Index outside the embedding table")
        out = np.empty(len(X), dtype=np.float64)
        for s in range(0, len(X), batch_size):
            x = self.embedding[X[s:s + batch_size]].reshape(-1, 2 * self.embedding.shape[1])
            for layer in self.dense:
                x = x @ layer.kernel
                if layer.bias is not None:
                    x = x + layer.bias
                x = layer.activation(x)
            out[s:s + batch_size] = x.reshape(-1)
        return out


# --------------------------------------------------------------------------- #
# Recommender = model + ID mapping + data
# --------------------------------------------------------------------------- #
class Recommender:
    def __init__(self, reader: WeightReader = h5py_reader):
        self.model = RatingModel(MODEL_PATH, reader=reader)
        self.user_ids = np.load(USER_IDS_PATH)     # u_encoder.classes_
        self.movie_ids = np.load(MOVIE_IDS_PATH)   # m_encoder.classes_
        self.n_users = len(self.user_ids)
        self.n_movies = len(self.movie_ids)
        if self.n_users + self.n_movies != self.model.input_dim:
            raise ValueError(
                f"ID mapping ({self.n_users} users + {self.n_movies} movies) does not match "
                f"the model's embedding size ({self.model.input_dim})")
        self._user_index = {int(u): i for i, u in enumerate(self.user_ids)}
        self._movie_index = {int(m): i for i, m in enumerate(self.movie_ids)}

        movies = pd.read_csv(MOVIES_PATH)
        movies["year"] = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
        movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        movies["genre_list"] = movies["genres"].fillna("").apply(
            lambda g: [] if g in ("", "(no genres listed)") else g.split("|"))
        self.ratings = pd.read_csv(RATINGS_PATH)
        pop = pd.read_csv(POPULARITY_PATH)  # number of ratings per movie in the training split

        # Candidate movies = movies the model saw during training (same rule the notebook
        # used to filter validation/test rows).
        cand = pop.merge(movies, on="movieId", how="left")
        cand = cand[cand["movieId"].isin(self._movie_index.keys())].copy()
        cand["movie_encoded"] = cand["movieId"].map(self._movie_index).astype(np.int64)
        self.candidates = cand.reset_index(drop=True)
        self.movies = movies
        self.all_genres = sorted({g for gl in cand["genre_list"].dropna() for g in gl})

    # --- helpers ----------------------------------------------------------- #
    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self._user_index

    def has_movie(self, movie_id: int) -> bool:
        return int(movie_id) in self._movie_index

    def _features(self, user_id: int, movie_encoded: np.ndarray) -> np.ndarray:
        """get_features() from the notebook: shift movie index by n_users."""
        u = self._user_index[int(user_id)]
        movie_encoded = np.asarray(movie_encoded, dtype=np.int64)
        return np.column_stack([np.full(len(movie_encoded), u, dtype=np.int64),
                                movie_encoded + self.n_users])

    def predict(self, user_id: int, movie_ids: Iterable[int]) -> np.ndarray:
        """Predicted ratings (clipped to 0.5-5.0) for one user and several movies."""
        if not self.has_user(user_id):
            raise KeyError(f"userId {user_id} was not in the training data")
        movie_ids = list(movie_ids)
        missing = [m for m in movie_ids if not self.has_movie(m)]
        if missing:
            raise KeyError(f"movieId(s) not in the training data: {missing[:5]}")
        enc = np.array([self._movie_index[int(m)] for m in movie_ids], dtype=np.int64)
        raw = self.model.predict_raw(self._features(user_id, enc))
        return np.clip(raw, RATING_MIN, RATING_MAX)

    def user_history(self, user_id: int) -> pd.DataFrame:
        hist = self.ratings[self.ratings["userId"] == int(user_id)]
        hist = hist.merge(self.movies[["movieId", "title", "genres", "year"]], on="movieId", how="left")
        known = hist["movieId"].isin(self._movie_index.keys())
        hist["predicted"] = np.nan
        if known.any():
            hist.loc[known, "predicted"] = self.predict(user_id, hist.loc[known, "movieId"])
        hist["date"] = pd.to_datetime(hist["timestamp"], unit="s").dt.date
        return hist.sort_values(["rating", "timestamp"], ascending=[False, False]).reset_index(drop=True)

    def recommend(self, user_id: int, n: int = 10, min_ratings: int = 1,
                  genres: Iterable[str] | None = None, exclude_seen: bool = True,
                  year_range: tuple[int, int] | None = None) -> pd.DataFrame:
        """Score every candidate movie for the user with the model and return the top-n."""
        if not self.has_user(user_id):
            raise KeyError(f"userId {user_id} was not in the training data")
        cand = self.candidates
        mask = cand["rating_count"] >= int(min_ratings)
        if exclude_seen:
            seen = set(self.ratings.loc[self.ratings["userId"] == int(user_id), "movieId"])
            mask &= ~cand["movieId"].isin(seen)
        genres = list(genres or [])
        if genres:
            gs = set(genres)
            mask &= cand["genre_list"].apply(lambda gl: bool(gs.intersection(gl)))
        if year_range is not None:
            lo, hi = year_range
            mask &= cand["year"].between(lo, hi)
        cand = cand[mask]
        if cand.empty:
            return cand.assign(predicted_rating=pd.Series(dtype=float)).head(0)
        raw = self.model.predict_raw(self._features(user_id, cand["movie_encoded"].to_numpy()))
        # Rank by the model's raw output; show the rating clipped to 0.5-5.0 as in the notebook.
        # (Ranking on the clipped value would turn every prediction above 5 into a tie.)
        out = cand.assign(model_score=raw, predicted_rating=np.clip(raw, RATING_MIN, RATING_MAX))
        out = out.sort_values(["model_score", "rating_count"], ascending=[False, False]).head(int(n))
        out.insert(0, "rank", range(1, len(out) + 1))
        return out.reset_index(drop=True)
