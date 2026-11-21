"""CineMatch: Streamlit front end for the MovieLens Deep Learning Recommendation Scoring Model.

All model, ranking and data logic lives in recommender.py (unchanged); the look lives in ui.py.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import ui
from recommender import Recommender

st.set_page_config(page_title="CineMatch · Movie Recommender", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the trained model and data…")
def load_recommender() -> Recommender:
    return Recommender()


def html(block: str) -> None:
    st.markdown(block, unsafe_allow_html=True)


try:
    rec = load_recommender()
except Exception as exc:  # show a clear message instead of a stack trace
    st.error(f"Could not load the model or data files: {exc}")
    st.stop()

user_list = [int(u) for u in rec.user_ids]

# ------------------------------------------------------------------ top navigation bar
nav_left, nav_right = st.columns([3, 1], vertical_alignment="center")
with nav_left:
    html(ui.nav())
with nav_right:
    user_id = st.selectbox("Viewer profile", user_list, index=0, format_func=lambda u: f"User #{u}",
                           help="Users the model was trained on (MovieLens userId 1–610).")

history = rec.user_history(user_id)
top_genre = history["genres"].str.split("|").explode()
top_genre = top_genre[top_genre != "(no genres listed)"]
fav_genre = top_genre.value_counts().index[0] if not top_genre.empty else "–"

tab_profile, tab_rec, tab_predict = st.tabs(["My Profile", "Discover", " Predict Rating"])

# ------------------------------------------------------------------ discover
with tab_rec:
    # What this user already loved, so the recommendations below have context
    top_rated = history.sort_values(["rating", "timestamp"], ascending=[False, False]).head(6)
    if not top_rated.empty:
        html(ui.section("Their Top Rated", f"Movies User #{user_id} rated highest"))
        html(ui.poster_grid(
            top_rated, "rating", "★ {:.1f}",
            sub_fn=lambda r: f"Rated {r['rating']:.1f}",
            overlay_fn=lambda r: [str(r["genres"]).replace("|", " · "), f"Rated on {r['date']}"]))

    html(ui.section("They May Also Like", "New picks ranked by predicted rating"))
    f1, f2, f3 = st.columns([3, 1.2, 1], vertical_alignment="bottom")
    with f1:
        genres = st.multiselect("Genres", rec.all_genres, default=[], placeholder="All genres")
    with f2:
        n_recs = st.select_slider("Titles to show", options=[6, 12, 18, 24, 30, 36, 42, 48], value=18)
    with f3:
        years = rec.candidates["year"].dropna()
        y_min, y_max = int(years.min()), int(years.max())
        with st.popover("⚙  More filters"):
            min_ratings = st.slider(
                "Minimum ratings per movie", 1, 100, 10,
                help="From movie_popularity.csv (ratings each movie had in the training set). "
                     "Movies with very few ratings have less reliable predictions.")
            year_range = st.slider("Release year", y_min, y_max, (y_min, y_max))
            exclude_seen = st.toggle("Hide movies this user already rated", value=True)

    recs = rec.recommend(user_id, n=n_recs, min_ratings=min_ratings, genres=genres,
                         exclude_seen=exclude_seen, year_range=year_range)
    if recs.empty:
        html(ui.empty("No titles match", "Try fewer genres, a lower minimum number of ratings, or a wider year range."))
    else:
        html(ui.hero(recs.iloc[0], user_id))
        html(ui.poster_grid(
            recs, "predicted_rating", "★ {:.2f}", ranked=True,
            sub_fn=lambda r: " · ".join(x for x in [ui._year(r), ", ".join(r["genre_list"][:2])] if x),
            overlay_fn=lambda r: [" · ".join(r["genre_list"][:3]), f"{int(r['rating_count'])} ratings in training",
                                  f"Raw model score {r['model_score']:.3f}"]))

# ------------------------------------------------------------------ profile
with tab_profile:
    valid = history.dropna(subset=["predicted"])
    mae = (valid["rating"] - valid["predicted"]).abs().mean() if not valid.empty else float("nan")
    html(ui.section(f"User #{user_id}", "Viewing history from ratings.csv"))
    html(ui.stats([
        ("Movies rated", f"{len(history):,}", "in the MovieLens data", False),
        ("Average rating", f"{history['rating'].mean():.2f}", "out of 5", True),
        ("Favourite genre", fav_genre, f"{int(top_genre.value_counts().iloc[0]) if not top_genre.empty else 0} titles", False),
        ("Model MAE on this user", f"{mae:.3f}", "includes training ratings", False),
    ]))

    favs = history.sort_values(["rating", "timestamp"], ascending=[False, False]).head(12).copy()
    if not favs.empty:
        html(ui.section("Highest rated", "This viewer's own top ratings"))
        html(ui.poster_grid(
            favs, "rating", "★ {:.1f}",
            sub_fn=lambda r: f"Rated {r['rating']:.1f} · model {r['predicted']:.2f}" if pd.notna(r["predicted"])
            else f"Rated {r['rating']:.1f}",
            overlay_fn=lambda r: [str(r["genres"]).replace("|", " · "), f"Rated on {r['date']}"]))

    html(ui.section("Taste profile"))
    left, right = st.columns(2)
    with left:
        st.caption("GENRES WATCHED")
        if not top_genre.empty:
            st.bar_chart(top_genre.value_counts().head(10), color=ui.GOLD, horizontal=True, height=320)
    with right:
        st.caption("RATING DISTRIBUTION")
        st.bar_chart(history["rating"].value_counts().sort_index(), color=ui.GOLD, height=320)

# ------------------------------------------------------------------ single prediction
with tab_predict:
    html(ui.section("Does it match their taste?", f"Pick any movie and see how well it fits User #{user_id}"))
    options = rec.candidates.sort_values("title")
    movie_titles = options["title"].tolist()
    movie_ids = options["movieId"].tolist()
    default_idx = movie_titles.index("Toy Story (1995)") if "Toy Story (1995)" in movie_titles else 0
    pick = st.selectbox("Search a movie", range(len(movie_titles)), index=default_idx,
                        format_func=lambda i: movie_titles[i], placeholder="Type a title…")
    movie_id = int(movie_ids[pick])
    pred = float(rec.predict(user_id, [movie_id])[0])
    actual = history.loc[history["movieId"] == movie_id, "rating"]
    st.write("")
    html(ui.predict_panel(options.iloc[pick], pred, float(actual.iloc[0]) if not actual.empty else None))

html(ui.footer())
