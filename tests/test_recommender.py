"""Checks that the app reproduces the notebook's results with the saved model.

Run:  python tests/test_recommender.py      (or: pytest tests)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import RATINGS_PATH, Recommender  # noqa: E402

_reader = None  # optional injected weight reader (used when h5py is unavailable)


def _rec():
    return Recommender() if _reader is None else Recommender(reader=_reader)


def test_notebook_test_metrics():
    """Rebuild the notebook's test split and compare MAE / RMSE (0.6743 / 0.8834)."""
    from sklearn.model_selection import train_test_split

    rec = _rec()
    df = pd.read_csv(RATINGS_PATH)[["userId", "movieId", "rating"]]
    train, tmp = train_test_split(df, test_size=0.2, random_state=42)
    _, test = train_test_split(tmp, test_size=0.5, random_state=42)
    test = test[test.userId.isin(set(train.userId)) & test.movieId.isin(set(train.movieId))]
    preds = np.concatenate([rec.predict(u, g.movieId) for u, g in test.groupby("userId", sort=False)])
    actual = np.concatenate([g.rating.to_numpy() for _, g in test.groupby("userId", sort=False)])
    mae = np.abs(preds - actual).mean()
    rmse = np.sqrt(((preds - actual) ** 2).mean())
    print(f"test MAE={mae:.6f} RMSE={rmse:.6f} rows={len(test)}")
    assert len(test) == 9695
    assert abs(mae - 0.674307) < 1e-4 and abs(rmse - 0.883368) < 1e-4


def test_notebook_sample_predictions():
    rec = _rec()
    titles = rec.movies.set_index("title")["movieId"]
    expected = [(491, "Meet the Fockers (2004)", 3.70), (603, "Apocalypse Now (1979)", 4.17),
                (412, "Alien (1979)", 4.09), (292, "Toy Story 3 (2010)", 4.17),
                (233, "The Shape of Water (2017)", 3.00)]
    for user, title, value in expected:
        got = round(float(rec.predict(user, [titles[title]])[0]), 2)
        assert got == value, (user, title, got, value)


def test_recommend_rules():
    rec = _rec()
    seen = set(rec.ratings.loc[rec.ratings.userId == 1, "movieId"])
    out = rec.recommend(1, n=20, min_ratings=10)
    assert len(out) == 20
    assert not set(out.movieId) & seen
    assert (out.rating_count >= 10).all()
    assert out.model_score.is_monotonic_decreasing
    assert out.predicted_rating.between(0.5, 5.0).all()
    # scores in the list equal single predictions
    assert np.allclose(out.predicted_rating, rec.predict(1, out.movieId))
    drama = rec.recommend(1, n=10, genres=["Drama"])
    assert drama.genre_list.apply(lambda g: "Drama" in g).all()
    assert rec.recommend(1, n=5, min_ratings=10**6).empty


def run_all():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)


if __name__ == "__main__":
    run_all()
