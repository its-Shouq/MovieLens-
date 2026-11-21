"""Front-end helpers: global CSS and HTML builders for the cinematic UI.

Presentation only. No model or recommendation logic lives here.
MovieLens has no poster images, so each poster is a designed title card. Its
colours come from the movie's main genre, and the same movie always gets the same card.
"""

from __future__ import annotations

import hashlib
import re
from html import escape

import pandas as pd

BG, BG2, CARD, GOLD, MUTED, TEXT = "#0B0B0D", "#17181C", "#222328", "#F5C518", "#A5A5A5", "#F5F5F5"

GENRE_TINT = {
    "Action": "#8C2A1E", "Adventure": "#8A5A17", "Animation": "#1E6A8C", "Children": "#2F7A4A",
    "Comedy": "#8C7417", "Crime": "#41414F", "Documentary": "#3E6658", "Drama": "#5A2E6E",
    "Fantasy": "#43308C", "Film-Noir": "#33333A", "Horror": "#6E1219", "IMAX": "#165E8C",
    "Musical": "#8C2F66", "Mystery": "#26365A", "Romance": "#8C2F43", "Sci-Fi": "#105A6B",
    "Thriller": "#2E4256", "War": "#565630", "Western": "#7A4C22",
}
DEFAULT_TINT = "#3A3B44"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --bg:{BG}; --bg2:{BG2}; --card:{CARD}; --gold:{GOLD}; --muted:{MUTED}; --text:{TEXT};
  --head:'Bebas Neue','Oswald','Impact',sans-serif; --body:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  --radius:14px; --ease:cubic-bezier(.2,.7,.2,1);
}}

/* ---------- base ---------- */
html, body, .stApp {{ background:var(--bg); color:var(--text); font-family:var(--body); }}
.stApp {{ background:
    radial-gradient(1200px 500px at 85% -10%, rgba(245,197,24,.06), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(120,120,160,.06), transparent 60%), var(--bg); }}
.stMarkdown, .stMarkdown p, label, input, textarea, button, [data-baseweb="select"], [data-baseweb="tag"] {{ font-family:var(--body); }}
h1, h2, h3, .cm-head {{ font-family:var(--head) !important; letter-spacing:.04em; font-weight:400 !important; }}

/* hide default chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ background:transparent; height:0; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding:1.1rem 2.4rem 2rem; max-width:1440px; }}

/* ---------- top nav ---------- */
.cm-nav {{ display:flex; align-items:center; gap:.9rem; padding:.35rem 0; }}
.cm-logo {{ font-family:var(--head); font-size:2.1rem; letter-spacing:.08em; color:var(--text); line-height:1; }}
.cm-logo span {{ color:var(--gold); }}
.cm-tag {{ color:var(--muted); font-size:.78rem; border-left:1px solid #2c2d33; padding-left:.9rem; line-height:1.3; }}

/* tabs styled as navigation */
[data-testid="stTabs"] [role="tablist"] {{ gap:.35rem; border-bottom:1px solid #222328; padding-bottom:.2rem; }}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display:none; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background:var(--gold); height:2px; }}
[data-testid="stTabs"] button[role="tab"] {{ background:transparent; border:none; padding:.55rem 1rem; border-radius:10px 10px 0 0;
  transition:color .25s var(--ease), background .25s var(--ease); }}
[data-testid="stTabs"] button[role="tab"] p {{ font-family:var(--body); font-weight:600; font-size:.92rem; color:var(--muted); letter-spacing:.02em; }}
[data-testid="stTabs"] button[role="tab"]:hover p {{ color:var(--text); }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {{ color:var(--gold); }}

/* ---------- inputs ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stNumberInput"] input {{
  background:var(--bg2) !important; border:1px solid #2a2b31 !important; border-radius:12px !important;
  transition:border-color .2s var(--ease), box-shadow .2s var(--ease); }}
[data-baseweb="select"] > div:hover {{ border-color:#3a3b42 !important; }}
[data-baseweb="select"] > div:focus-within {{ border-color:var(--gold) !important; box-shadow:0 0 0 3px rgba(245,197,24,.12); }}
[data-baseweb="tag"] {{ background:rgba(245,197,24,.14) !important; border-radius:999px !important; }}
[data-baseweb="tag"] span {{ color:var(--gold) !important; }}
[data-baseweb="popover"] li[role="option"]:hover, [data-baseweb="popover"] li[aria-selected="true"] {{ background:rgba(245,197,24,.1) !important; }}
label, [data-testid="stWidgetLabel"] p {{ color:var(--muted) !important; font-size:.78rem !important; font-weight:600 !important;
  text-transform:uppercase; letter-spacing:.08em; }}
[data-testid="stSlider"] [role="slider"] {{ background:var(--gold); box-shadow:0 0 0 4px rgba(245,197,24,.18); }}

/* ---------- buttons ---------- */
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button {{
  font-family:var(--body); font-weight:600; border-radius:999px; padding:.5rem 1.2rem;
  background:var(--bg2); color:var(--text); border:1px solid #2f3036;
  transition:transform .2s var(--ease), border-color .2s var(--ease), background .2s var(--ease); }}
.stButton button:hover, .stDownloadButton button:hover, [data-testid="stPopover"] button:hover {{
  border-color:var(--gold); color:var(--gold); transform:translateY(-1px); }}
.stDownloadButton button {{ background:var(--gold); color:#111 !important; border-color:var(--gold); }}
.stDownloadButton button:hover {{ background:#ffd43b; color:#111 !important; }}
.stDownloadButton button p {{ color:#111 !important; }}

/* expanders / dataframes / alerts */
[data-testid="stExpander"] details {{ background:var(--bg2); border:1px solid #24252b; border-radius:var(--radius); }}
[data-testid="stExpander"] summary p {{ font-weight:600; }}
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; border:1px solid #24252b; }}
[data-testid="stAlert"] {{ border-radius:12px; }}

/* ---------- hero ---------- */
.cm-hero {{ position:relative; overflow:hidden; border-radius:22px; margin:1.1rem 0 1.6rem; min-height:360px;
  display:grid; grid-template-columns:minmax(0,1fr) 230px; gap:2.2rem; align-items:center; padding:2.6rem 3rem;
  border:1px solid #23242a; box-shadow:0 30px 80px -30px rgba(0,0,0,.8); }}
.cm-hero::after {{ content:""; position:absolute; inset:0; background:
  linear-gradient(90deg, rgba(11,11,13,.96) 0%, rgba(11,11,13,.75) 45%, rgba(11,11,13,.2) 100%),
  linear-gradient(0deg, rgba(11,11,13,.85), transparent 45%); z-index:0; }}
.cm-hero > * {{ position:relative; z-index:1; }}
.cm-eyebrow {{ display:inline-flex; align-items:center; gap:.5rem; color:var(--gold); font-weight:600; font-size:.75rem;
  letter-spacing:.22em; text-transform:uppercase; }}
.cm-eyebrow::before {{ content:""; width:26px; height:2px; background:var(--gold); display:inline-block; }}
.cm-hero h1 {{ font-size:clamp(2.6rem, 5.4vw, 4.8rem); line-height:.95; margin:.6rem 0 .8rem; color:var(--text); }}
.cm-meta {{ display:flex; flex-wrap:wrap; gap:.55rem; align-items:center; color:var(--muted); font-size:.9rem; }}
.cm-dot {{ width:4px; height:4px; border-radius:50%; background:#55565c; display:inline-block; }}
.cm-score {{ display:flex; align-items:baseline; gap:.5rem; margin-top:1.3rem; }}
.cm-score b {{ font-family:var(--head); font-size:3.2rem; color:var(--gold); font-weight:400; line-height:1; }}
.cm-score span {{ color:var(--muted); font-size:.85rem; }}
.cm-hero p.cm-blurb {{ color:#cfcfcf; max-width:560px; font-size:.95rem; line-height:1.6; margin-top:1rem; }}
.cm-hero .cm-poster {{ width:230px; box-shadow:0 30px 60px -20px rgba(0,0,0,.9); transform:rotate(1.5deg); }}

/* ---------- section header ---------- */
.cm-section {{ display:flex; flex-wrap:wrap; align-items:end; justify-content:space-between; gap:1rem; margin:1.6rem 0 .9rem; }}
.cm-section h2 {{ font-size:2rem; margin:0; color:var(--text); }}
.cm-section small {{ color:var(--muted); font-size:.85rem; }}

/* ---------- poster grid ---------- */
.cm-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(178px, 1fr)); gap:1.35rem 1.1rem; }}
.cm-card {{ position:relative; transition:transform .35s var(--ease); }}
.cm-card:hover {{ transform:translateY(-6px); }}
.cm-poster {{ position:relative; aspect-ratio:2/3; border-radius:var(--radius); overflow:hidden; isolation:isolate;
  box-shadow:0 14px 30px -14px rgba(0,0,0,.85); border:1px solid rgba(255,255,255,.06);
  transition:box-shadow .35s var(--ease), border-color .35s var(--ease); }}
.cm-card:hover .cm-poster {{ box-shadow:0 26px 50px -18px rgba(0,0,0,.95), 0 0 0 1px rgba(245,197,24,.55); border-color:rgba(245,197,24,.4); }}
.cm-poster::before {{ content:""; position:absolute; inset:0; z-index:-1;
  background:repeating-linear-gradient(115deg, rgba(255,255,255,.025) 0 2px, transparent 2px 9px); }}
.cm-poster .p-title {{ position:absolute; left:14px; right:14px; bottom:32px; font-family:var(--head); color:#fff;
  font-size:1.65rem; line-height:.98; letter-spacing:.03em; text-shadow:0 2px 14px rgba(0,0,0,.6);
  display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient:vertical; overflow:hidden; }}
.cm-poster .p-year {{ position:absolute; left:14px; top:14px; color:rgba(255,255,255,.75); font-size:.72rem; font-weight:600; letter-spacing:.14em; }}
.cm-poster .p-genre {{ position:absolute; left:14px; right:14px; bottom:14px; color:rgba(255,255,255,.6); font-size:.62rem;
  letter-spacing:.16em; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.cm-poster .p-mark {{ position:absolute; right:-10px; top:28%; font-family:var(--head); font-size:9rem; line-height:1; color:rgba(255,255,255,.06); }}
.cm-badge {{ position:absolute; right:10px; top:10px; z-index:2; background:rgba(11,11,13,.78); backdrop-filter:blur(6px);
  color:var(--gold); font-weight:700; font-size:.78rem; padding:.28rem .55rem; border-radius:999px; border:1px solid rgba(245,197,24,.35); }}
.cm-rank {{ flex:none; font-family:var(--head); font-size:2.5rem; line-height:.9; color:transparent;
  -webkit-text-stroke:1.2px rgba(245,197,24,.9); min-width:1.1ch; }}
.cm-info {{ display:flex; gap:.55rem; align-items:flex-start; }}
.cm-info > div:last-child {{ min-width:0; flex:1; }}
.cm-overlay {{ position:absolute; inset:0; z-index:2; display:flex; flex-direction:column; justify-content:flex-end; gap:.35rem;
  padding:14px; background:linear-gradient(0deg, rgba(11,11,13,.96) 15%, rgba(11,11,13,.55) 60%, transparent);
  opacity:0; transition:opacity .3s var(--ease); }}
.cm-card:hover .cm-overlay {{ opacity:1; }}
.cm-overlay .o-score {{ font-family:var(--head); font-size:2.2rem; color:var(--gold); line-height:1; }}
.cm-overlay .o-line {{ color:#d6d6d6; font-size:.76rem; line-height:1.35; }}
.cm-info {{ padding:.75rem .15rem 0 .25rem; }}
.cm-info .i-title {{ font-weight:600; font-size:.9rem; color:var(--text); line-height:1.3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.cm-info .i-sub {{ color:var(--muted); font-size:.76rem; margin-top:.15rem; }}

/* ---------- stat tiles ---------- */
.cm-stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:.9rem; margin:.4rem 0 1rem; }}
.cm-stat {{ background:linear-gradient(180deg, var(--card), var(--bg2)); border:1px solid #2a2b31; border-radius:var(--radius); padding:1rem 1.15rem; }}
.cm-stat .s-label {{ color:var(--muted); font-size:.7rem; font-weight:600; letter-spacing:.14em; text-transform:uppercase; }}
.cm-stat .s-value {{ font-family:var(--head); font-size:2.2rem; color:var(--text); line-height:1.05; margin-top:.25rem; }}
.cm-stat .s-value.gold {{ color:var(--gold); }}
.cm-stat .s-note {{ color:var(--muted); font-size:.74rem; }}

/* ---------- panels ---------- */
.cm-panel {{ background:linear-gradient(180deg, var(--card), var(--bg2)); border:1px solid #2a2b31; border-radius:18px; padding:1.4rem 1.6rem; }}
.cm-panel h3 {{ font-size:1.5rem; margin:0 0 .6rem; color:var(--text); }}
.cm-panel p, .cm-panel li {{ color:#cfcfcf; font-size:.92rem; line-height:1.65; }}
.cm-panel code {{ background:#2c2d33; color:var(--gold); padding:.1rem .35rem; border-radius:6px; font-size:.82rem; }}
.cm-steps {{ counter-reset:s; list-style:none; padding:0; margin:0; }}
.cm-steps li {{ counter-increment:s; position:relative; padding-left:2.6rem; margin-bottom:.9rem; }}
.cm-steps li::before {{ content:counter(s, decimal-leading-zero); position:absolute; left:0; top:-.1rem; font-family:var(--head);
  font-size:1.5rem; color:var(--gold); }}
.cm-predict {{ display:grid; grid-template-columns:220px minmax(0,1fr); gap:2rem; align-items:center; }}
.cm-predict h2 {{ font-size:clamp(2rem, 4vw, 3.2rem); margin:.4rem 0 .6rem; line-height:1; }}
.cm-meter {{ height:8px; background:#2c2d33; border-radius:999px; overflow:hidden; margin:.6rem 0 .3rem; max-width:420px; }}
.cm-meter > div {{ height:100%; background:linear-gradient(90deg, #b8900f, var(--gold)); border-radius:999px; }}
.cm-pill {{ display:inline-block; padding:.25rem .7rem; border-radius:999px; background:#2a2b31; color:#d8d8d8; font-size:.74rem; margin:0 .35rem .35rem 0; }}
.cm-empty {{ text-align:center; padding:3rem 1rem; color:var(--muted); border:1px dashed #2f3036; border-radius:18px; }}
.cm-empty b {{ font-family:var(--head); font-size:1.8rem; color:var(--text); letter-spacing:.04em; display:block; }}

/* ---------- footer ---------- */
.cm-footer {{ margin-top:3.5rem; padding:1.8rem 0 .6rem; border-top:1px solid #222328; display:flex; flex-wrap:wrap;
  justify-content:space-between; gap:1rem; color:var(--muted); font-size:.8rem; }}
.cm-footer .cm-logo {{ font-size:1.5rem; }}
.cm-footer a {{ color:var(--muted); text-decoration:none; border-bottom:1px solid #3a3b42; }}
.cm-footer a:hover {{ color:var(--gold); border-color:var(--gold); }}

/* ---------- responsive ---------- */
@media (max-width: 900px) {{
  .block-container, [data-testid="stMainBlockContainer"] {{ padding:.8rem 1rem 2rem; }}
  .cm-hero {{ grid-template-columns:1fr; padding:1.8rem 1.4rem; min-height:auto; }}
  .cm-hero .cm-poster {{ display:none; }}
  .cm-tag {{ display:none; }}
  .cm-predict {{ grid-template-columns:1fr; }}
  .cm-predict .cm-poster {{ max-width:200px; }}
}}
@media (max-width: 520px) {{
  .cm-grid {{ grid-template-columns:repeat(2, minmax(0, 1fr)); gap:1.1rem .8rem; }}
  .cm-poster .p-title {{ font-size:1.3rem; }}
  [data-testid="stTabs"] button[role="tab"] {{ padding:.5rem .6rem; }}
}}
</style>
"""


# --------------------------------------------------------------------------- helpers
def _e(x) -> str:
    return escape(str(x), quote=True)


def _genres(row) -> list[str]:
    gl = row.get("genre_list") if hasattr(row, "get") else None
    if isinstance(gl, list):
        return gl
    g = str(row.get("genres", "") or "")
    return [] if g in ("", "(no genres listed)", "nan") else g.split("|")


def _year(row) -> str:
    y = row.get("year")
    return "" if y is None or pd.isna(y) else str(int(y))


def _title(row) -> str:
    t = row.get("clean_title")
    if t is None or (isinstance(t, float) and pd.isna(t)):
        t = str(row.get("title", "")).rsplit(" (", 1)[0]
    t = str(t)
    # MovieLens style "Godfather, The" -> "The Godfather"
    for art in ("The", "A", "An"):
        if t.endswith(f", {art}"):
            t = f"{art} {t[: -len(art) - 2]}"
    return t


def poster_style(title: str, genres: list[str]) -> str:
    tint = GENRE_TINT.get(genres[0], DEFAULT_TINT) if genres else DEFAULT_TINT
    tint2 = GENRE_TINT.get(genres[1], tint) if len(genres) > 1 else tint
    h = int(hashlib.md5(title.encode("utf-8")).hexdigest()[:6], 16)
    angle, x, y = 140 + h % 60, 20 + (h >> 4) % 60, 10 + (h >> 9) % 40
    return (f"background:radial-gradient(120% 70% at {x}% {y}%, {tint2}cc 0%, transparent 60%),"
            f"linear-gradient({angle}deg, {tint} 0%, #141418 78%);")


def _short(title: str) -> str:
    short = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    return short or title


def poster(row, badge: str | None = None, extra_class: str = "") -> str:
    title, genres, year = _short(_title(row)), _genres(row), _year(row)
    mark = _e(title[:1].upper() if title else "")
    b = f"<div class='cm-badge'>{_e(badge)}</div>" if badge else ""
    return (f"<div class='cm-poster {extra_class}' style=\"{poster_style(title, genres)}\">{b}"
            f"<div class='p-mark'>{mark}</div><div class='p-year'>{_e(year)}</div>"
            f"<div class='p-genre'>{_e(' · '.join(genres[:2]))}</div>"
            f"<div class='p-title'>{_e(title)}</div></div>")


# --------------------------------------------------------------------------- blocks
def nav() -> str:
    return ("<div class='cm-nav'><div class='cm-logo'>CINE<span>MATCH</span></div>"
            "<div class='cm-tag'>Deep learning movie recommendations<br>trained on MovieLens</div></div>")


def hero(row, user_id: int) -> str:
    genres = _genres(row)
    meta = [x for x in (_year(row), " · ".join(genres[:3]), f"{int(row['rating_count'])} ratings in training") if x]
    meta_html = "<span class='cm-dot'></span>".join(f"<span>{_e(m)}</span>" for m in meta)
    return (f"<div class='cm-hero' style=\"{poster_style(_title(row), genres)}\">"
            f"<div><div class='cm-eyebrow'>Top pick for user #{user_id}</div>"
            f"<h1>{_e(_short(_title(row)))}</h1><div class='cm-meta'>{meta_html}</div>"
            f"<div class='cm-score'><b>{row['predicted_rating']:.2f}</b><span>/ 5 &nbsp;predicted rating</span></div>"
            f"<p class='cm-blurb'>Of all the movies matching your filters, this is the one our neural "
            f"network expects them to rate highest. It was scored from the user's and the movie's learned embeddings.</p></div>"
            f"{poster(row)}</div>")


def section(title: str, note: str = "") -> str:
    return f"<div class='cm-section'><h2>{_e(title)}</h2><small>{_e(note)}</small></div>"


def poster_grid(df: pd.DataFrame, score_col: str, badge_fmt: str, ranked: bool = False,
                sub_fn=None, overlay_fn=None) -> str:
    cards = []
    for _, row in df.iterrows():
        score = row[score_col]
        rank = f"<div class='cm-rank'>{int(row['rank'])}</div>" if ranked and "rank" in row else ""
        overlay = ""
        if overlay_fn:
            lines = "".join(f"<div class='o-line'>{_e(t)}</div>" for t in overlay_fn(row))
            overlay = f"<div class='cm-overlay'><div class='o-score'>★ {score:.2f}</div>{lines}</div>"
        sub = sub_fn(row) if sub_fn else _year(row)
        inner = poster(row, badge=badge_fmt.format(score))
        inner = inner[:-len("</div>")] + overlay + "</div>"
        cards.append(f"<div class='cm-card'>{inner}<div class='cm-info'>{rank}<div>"
                     f"<div class='i-title' title=\"{_e(_title(row))}\">{_e(_title(row))}</div>"
                     f"<div class='i-sub'>{_e(sub)}</div></div></div></div>")
    return f"<div class='cm-grid{' ranked' if ranked else ''}'>{''.join(cards)}</div>"


def stats(items: list[tuple[str, str, str, bool]]) -> str:
    tiles = "".join(
        f"<div class='cm-stat'><div class='s-label'>{_e(l)}</div>"
        f"<div class='s-value{' gold' if g else ''}'>{_e(v)}</div><div class='s-note'>{_e(n)}</div></div>"
        for l, v, n, g in items)
    return f"<div class='cm-stats'>{tiles}</div>"


def predict_panel(row, pred: float, actual: float | None) -> str:
    genres = "".join(f"<span class='cm-pill'>{_e(g)}</span>" for g in _genres(row))
    actual_txt = f"{actual:.1f} / 5" if actual is not None else "Not rated yet"
    meta = " · ".join(x for x in (_year(row), f"{int(row['rating_count'])} ratings in training") if x)
    # Predicted rating (0.5–5) shown as a percentage, like a streaming "match" score
    match = max(0.0, min(100.0, (pred - 0.5) / 4.5 * 100))
    return (f"<div class='cm-panel cm-predict'>{poster(row)}<div>"
            f"<div class='cm-eyebrow'>Taste match</div><h2>{_e(_title(row))}</h2>"
            f"<div class='cm-meta'><span>{_e(meta)}</span></div><div style='margin:.8rem 0'>{genres}</div>"
            f"<div class='cm-score'><b>{match:.0f}%</b><span>match for their taste</span></div>"
            f"<div class='cm-meter'><div style='width:{match:.1f}%'></div></div>"
            f"<div class='cm-meta'><span>Predicted rating: <b style='color:#F5F5F5'>{pred:.2f} / 5</b></span>"
            f"<span class='cm-dot'></span>"
            f"<span>Actual rating: <b style='color:#F5F5F5'>{_e(actual_txt)}</b></span></div>"
            f"</div></div>")


def empty(title: str, text: str) -> str:
    return f"<div class='cm-empty'><b>{_e(title)}</b>{_e(text)}</div>"


def footer() -> str:
    return ("<div class='cm-footer'><div><div class='cm-logo'>CINE<span>MATCH</span></div>"
            "<div>Deep Learning Recommendation Scoring Model</div></div>"
            "<div style='text-align:right'>Data: <a href='https://grouplens.org/datasets/movielens/' target='_blank'>"
            "MovieLens, GroupLens Research</a><br>Embedding + MLP · Test MAE 0.674 · Built with Streamlit</div></div>")
