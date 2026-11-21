<<<<<<< HEAD
# MovieLens-
=======
# 🎬 CineMatch: MovieLens Movie Recommender (Deep Learning), Streamlit App

A web app for my **Deep Learning Recommendation Scoring Model** trained on MovieLens
(`MovieLens_DL2_project.ipynb`). For a selected user, the trained network predicts a rating for every
movie seen in training, and the highest predictions become recommendations.

**Features** (dark cinematic UI: Bebas Neue + Inter, golden accents, responsive poster grid)
- 👤 **My Profile**: the user's highest-rated movies as posters, favourite genres, rating distribution
- 🔍 **Discover**: *Their Top Rated* (the user's favourite movies), then *They May Also Like*: a hero banner for the top pick and top‑N movie cards with predicted ratings, filtered by genre, release year and minimum popularity
- 🎯 **Predict Rating**: a *Taste Match %* for any user–movie pair, with the predicted rating and the user's actual rating if they rated it

## Model (unchanged from the notebook)
| Item | Value |
|---|---|
| Input | `[user_index, movie_index + 610]` (LabelEncoder indices) |
| Layers | Embedding(10334, 32) → Flatten → Dense 128 ReLU → Dense 64 ReLU → Dense 32 ReLU → Dense 1 linear |
| Training | Adam lr = 1e‑5, batch 32, L2 = 0.001, early stopping on val MAE |
| Test MAE / RMSE | **0.6743 / 0.8834** (mean‑rating baseline: 0.8306 / 1.0502) |
| Output | predicted rating clipped to 0.5–5.0 |

The model is **not retrained**. `recommender.py` reads the trained weights from `models/final_model.keras`
and runs the network's forward pass with NumPy (the same operations as `model.predict`). This means the
deployed app doesn't need TensorFlow, which is very large and slow to install on Streamlit Community Cloud.
`tests/test_recommender.py` rebuilds the notebook's test split and reproduces **Test MAE 0.674307**
and the notebook's sample predictions. `scripts/verify_with_keras.py` compares the app's output with Keras directly.

## Project structure
```
movie-recommender-app/
├── app.py                     # Streamlit page (layout + widgets)
├── ui.py                      # cinematic theme: CSS + poster/hero/card HTML
├── recommender.py             # model loading + inference + recommendation logic
├── requirements.txt
├── .streamlit/config.toml     # theme + server settings
├── models/
│   ├── final_model.keras      # trained model (config + weights)
│   ├── user_ids.npy           # u_encoder.classes_ (610 users)
│   └── movie_ids.npy          # m_encoder.classes_ (9,724 movies)
├── data/
│   ├── ratings.csv            # MovieLens ratings (user history, "already rated" filter)
│   ├── movies.csv             # titles + genres
│   └── movie_popularity.csv   # ratings per movie in the training split
├── tests/test_recommender.py  # reproduces the notebook's results
└── scripts/verify_with_keras.py
```

## Run locally
```bash
cd movie-recommender-app
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501 (if that port is taken, Streamlit picks the next one, e.g. 8503. Use the *Local URL* it prints).

Optional checks:
```bash
pip install scikit-learn pytest
python tests/test_recommender.py            # expects: test MAE=0.674307 ... PASS x3

pip install tensorflow keras==3.13.2        # only for this check, not needed by the app
python scripts/verify_with_keras.py         # expects: OK - the app reproduces the Keras model's predictions.
```

## Deploy: step 1, push to GitHub
1. Go to https://github.com/new and create a **public** repository, e.g. `movie-recommender-app`. Don't add a README (this project already has one).
2. In a terminal, inside the `movie-recommender-app` folder:
   ```bash
   git init
   git add .
   git commit -m "Streamlit movie recommender"
   git branch -M main
   git remote add origin https://github.com/<your-username>/movie-recommender-app.git
   git push -u origin main
   ```
   (No terminal? On the new repository page click **uploading an existing file** and drag in *all*
   files and folders. Make sure `.streamlit/config.toml` is included, because hidden folders are easy to miss.)
3. Check on GitHub that `app.py`, `requirements.txt`, `models/` and `data/` are at the top level of the repo.
   Every file is under 5 MB, so Git LFS isn't needed.

## Deploy: step 2, Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub (allow access to your repositories).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/movie-recommender-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose a name, e.g. `reema-movie-recommender`
4. Open **Advanced settings** → **Python version: 3.12** → Save.
5. Click **Deploy**. The first build installs `requirements.txt` and takes 1–3 minutes.
6. When the app loads, select a user and check the recommendations. Share the `https://<name>.streamlit.app` link.

**Updating:** push new commits to `main` and the app redeploys automatically.

## Troubleshooting
| Problem | Fix |
|---|---|
| `Could not load the model or data files` | Make sure `models/` and `data/` were pushed with the exact file names above. |
| `ModuleNotFoundError` | `requirements.txt` must be in the repo root, next to `app.py`. |
| App sleeps after inactivity | Normal on the free tier. Click **Yes, get this app back up!** |
| Want to use your original `final_model.keras` | Replace `models/final_model.keras` with it (same name). |

## Limitations
- MovieLens has no poster images, so each poster is a designed title card coloured by the movie's genre.
- The model only knows the 610 users and 8,983 movies seen in training, so new users can't be scored (no embedding exists for them).
- Movies with very few ratings get less reliable predictions; raise *Minimum number of ratings per movie* to focus on well-known titles.

Dataset: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* (https://grouplens.org/datasets/movielens/)
>>>>>>> f140603 (Add CineMatch demo)
