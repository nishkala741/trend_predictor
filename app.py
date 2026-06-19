"""
╔══════════════════════════════════════════════════════════════════╗
║   FASHION-TECH · Retail Intelligence Dashboard                   ║
║   Stage 3 — Interactive Streamlit Application                    ║
║                                                                  ║
║   Run locally :  streamlit run app.py                            ║
║   Data file   :  processed_fashion_reviews.csv  (same directory) ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────
import re
import warnings
from collections import Counter

# ── Third-party ───────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# 0.  PAGE CONFIG  (must be the very first Streamlit call)
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fashion-Tech · Retail Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# 1.  GLOBAL DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════
PALETTE = {
    "bg"         : "#0d0d14",
    "surface"    : "#16161f",
    "surface2"   : "#1e1e2e",
    "border"     : "#2a2a3d",
    "accent"     : "#c084fc",      # soft violet — fashion-forward
    "accent2"    : "#f472b6",      # hot pink highlight
    "positive"   : "#34d399",
    "negative"   : "#f87171",
    "neutral"    : "#94a3b8",
    "text"       : "#e2e8f0",
    "text_muted" : "#64748b",
}

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font         =dict(family="Inter, sans-serif", color=PALETTE["text"]),
        colorway     =[PALETTE["accent"], PALETTE["accent2"],
                       PALETTE["positive"], PALETTE["negative"],
                       "#60a5fa", "#fbbf24"],
        xaxis        =dict(gridcolor=PALETTE["border"], showgrid=True),
        yaxis        =dict(gridcolor=PALETTE["border"], showgrid=True),
        margin       =dict(l=20, r=20, t=40, b=20),
    )
)

# ── Inject global CSS ─────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Base ─────────────────────────────── */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color : {PALETTE["bg"]};
      color            : {PALETTE["text"]};
      font-family      : 'Inter', 'Segoe UI', sans-serif;
  }}
  [data-testid="stSidebar"] {{
      background-color : {PALETTE["surface"]};
      border-right     : 1px solid {PALETTE["border"]};
  }}
  [data-testid="stSidebar"] .stMarkdown p {{
      color : {PALETTE["text_muted"]};
      font-size : 0.78rem;
      text-transform : uppercase;
      letter-spacing : 0.08em;
  }}

  /* ── KPI cards ────────────────────────── */
  .kpi-card {{
      background    : {PALETTE["surface2"]};
      border        : 1px solid {PALETTE["border"]};
      border-radius : 14px;
      padding       : 22px 24px;
      text-align    : center;
  }}
  .kpi-label {{
      font-size      : 0.72rem;
      text-transform : uppercase;
      letter-spacing : 0.1em;
      color          : {PALETTE["text_muted"]};
      margin-bottom  : 6px;
  }}
  .kpi-value {{
      font-size   : 2.1rem;
      font-weight : 700;
      color       : {PALETTE["accent"]};
      line-height : 1;
  }}
  .kpi-sub {{
      font-size  : 0.75rem;
      color      : {PALETTE["text_muted"]};
      margin-top : 4px;
  }}

  /* ── Section headers ──────────────────── */
  .section-header {{
      font-size      : 0.7rem;
      font-weight    : 600;
      text-transform : uppercase;
      letter-spacing : 0.14em;
      color          : {PALETTE["accent"]};
      border-bottom  : 1px solid {PALETTE["border"]};
      padding-bottom : 6px;
      margin-bottom  : 16px;
  }}

  /* ── Sentiment tester ─────────────────── */
  .result-box {{
      background    : {PALETTE["surface2"]};
      border-radius : 12px;
      padding       : 20px 24px;
      margin-top    : 12px;
      border-left   : 4px solid {PALETTE["accent"]};
  }}
  .result-label {{
      font-size   : 1.6rem;
      font-weight : 700;
  }}
  .positive-label {{ color : {PALETTE["positive"]}; }}
  .negative-label {{ color : {PALETTE["negative"]}; }}

  /* ── Misc ────────────────────────────── */
  .stTextArea textarea {{
      background-color : {PALETTE["surface2"]} !important;
      color            : {PALETTE["text"]}     !important;
      border           : 1px solid {PALETTE["border"]} !important;
      border-radius    : 10px !important;
  }}
  .stSlider [data-baseweb="slider"] {{ color : {PALETTE["accent"]}; }}
  div[data-testid="stMetricValue"] {{ color: {PALETTE["accent"]}; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 2.  DATA LAYER
# ═══════════════════════════════════════════════════════════════════

FEATURE_KEYWORDS = {
    "Fabric / Material" : ["fabric", "material", "cotton", "polyester", "silk",
                            "texture", "cloth", "quality", "feel", "thin", "cheap"],
    "Sizing"            : ["size", "sizing", "fit", "fits", "small", "large",
                            "tight", "loose", "true to size", "xl", "xs"],
    "Colour / Print"    : ["colour", "color", "faded", "print", "pattern",
                            "shade", "dye", "bleed"],
    "Stitching"         : ["stitch", "stitching", "seam", "tear", "loose thread",
                            "frayed", "unravel"],
    "Delivery / Pack"   : ["delivery", "packaging", "damaged", "late", "missing",
                            "wrong item", "courier"],
    "Value for Money"   : ["price", "expensive", "overpriced", "worth", "value",
                            "money", "cost", "cheap for price"],
}

STOP_WORDS = set([
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "into", "about", "this", "that",
    "these", "those", "so", "very", "just", "also", "not", "no", "nor",
    "bought", "got", "get", "like", "even", "really", "quite", "than",
])


def _make_mock_data(n: int = 600) -> pd.DataFrame:
    """
    Generate a realistic mock dataset so the dashboard is fully
    interactive even when the real CSV is absent.
    """
    rng = np.random.default_rng(42)

    tiers   = ["Budget", "Value", "Mid-Range", "Premium", "Luxury"]
    pop     = ["Niche (<100)", "Growing (100-999)", "Popular (1K-5K)",
               "Trending (5K-10K)", "Viral (10K+)"]
    sellers = ["Roadster", "DILLINGER", "H&M", "Zara", "Mango",
               "W", "AND", "BIBA", "Aurelia", "Global Desi"]

    pos_snippets = [
        "The fabric quality is outstanding and the fit is perfect.",
        "Loved the colour and the stitching is very neat.",
        "Excellent value for money, would definitely buy again.",
        "Arrived quickly, packaging was great, very happy.",
        "The material feels premium and drapes beautifully.",
    ]
    neg_snippets = [
        "Fabric feels very cheap and thin for the price paid.",
        "The stitching came apart after one wash, very disappointed.",
        "Size runs too small, not true to size at all.",
        "Colour faded significantly after the first wash.",
        "Overpriced for the quality, expected much better material.",
        "Delivery was late and the packaging was damaged.",
    ]

    tier_idx    = rng.integers(0, len(tiers),   size=n)
    pop_idx     = rng.integers(0, len(pop),     size=n)
    seller_idx  = rng.integers(0, len(sellers), size=n)

    base_rating = 2.8 + tier_idx * 0.28 + rng.normal(0, 0.5, size=n)
    rating      = np.clip(base_rating, 1.0, 5.0).round(1)

    bert_score  = np.where(
        rating >= 3.5,
        rng.uniform(0.72, 0.99, size=n),
        rng.uniform(0.52, 0.89, size=n)
    )
    bert_label  = np.where(rating >= 3.5, "POSITIVE", "NEGATIVE")

    review_text = [
        pos_snippets[rng.integers(0, len(pos_snippets))]
        if lbl == "POSITIVE"
        else neg_snippets[rng.integers(0, len(neg_snippets))]
        for lbl in bert_label
    ]

    return pd.DataFrame({
        "name"           : [f"Product_{i:04d}" for i in range(n)],
        "Price_Tier"     : [tiers[i]   for i in tier_idx],
        "Popularity_Bin" : [pop[i]     for i in pop_idx],
        "seller"         : [sellers[i] for i in seller_idx],
        "rating"         : rating,
        "BERT_Label"     : bert_label,
        "BERT_Score"     : bert_score.round(4),
        "review_text"    : review_text,
        "Age"            : rng.integers(18, 65, size=n),
    })


@st.cache_data(show_spinner=False)
def load_data(csv_path: str = "processed_fashion_reviews.csv"):
    """
    Load processed CSV from Stage 2.
    Returns (dataframe, is_mock).
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        if "BERT_Label" not in df.columns and "Rating_Sentiment" in df.columns:
            df["BERT_Label"] = df["Rating_Sentiment"].str.upper()
        if "BERT_Score" not in df.columns and "rating" in df.columns:
            df["BERT_Score"] = (df["rating"] / 5.0).clip(0.5, 1.0)
        if "review_text" not in df.columns:
            df["review_text"] = df.get("name", "Unknown product").astype(str)
        if "Age" not in df.columns:
            rng = np.random.default_rng(0)
            df["Age"] = rng.integers(18, 65, size=len(df))

        df["BERT_Score"] = pd.to_numeric(df["BERT_Score"], errors="coerce").fillna(0.5)
        df["rating"]     = pd.to_numeric(
            df.get("rating", pd.Series(3.0, index=df.index)),
            errors="coerce"
        ).fillna(3.0)
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(30).astype(int)

        return df, False

    except FileNotFoundError:
        return _make_mock_data(), True


# ═══════════════════════════════════════════════════════════════════
# 3.  SENTIMENT ENGINE  (DistilBERT -> TextBlob -> keyword fallback)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """
    Try DistilBERT first, then TextBlob, then keyword heuristic.
    Returns (predict_fn, model_name_str).
    """
    # Attempt 1: DistilBERT
    try:
        import torch
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        pipe   = hf_pipeline(
            "text-classification",
            model      = "distilbert-base-uncased-finetuned-sst-2-english",
            device     = device,
            truncation = True,
            max_length = 128,
        )

        def predict_distilbert(text: str) -> dict:
            result = pipe(text[:512])[0]
            return {
                "label": result["label"],
                "score": round(result["score"], 4),
                "model": "DistilBERT (SST-2)"
            }

        return predict_distilbert, "DistilBERT · distilbert-base-uncased-finetuned-sst-2"

    except Exception:
        pass

    # Attempt 2: TextBlob
    try:
        from textblob import TextBlob

        def predict_textblob(text: str) -> dict:
            polarity = TextBlob(text).sentiment.polarity
            if polarity >= 0:
                label = "POSITIVE"
                score = round(0.5 + polarity * 0.5, 4)
            else:
                label = "NEGATIVE"
                score = round(0.5 - polarity * 0.5, 4)
            return {"label": label, "score": score, "model": "TextBlob (statistical fallback)"}

        return predict_textblob, "TextBlob (offline statistical fallback)"

    except Exception:
        pass

    # Attempt 3: Keyword heuristic
    POS = {"good", "great", "love", "excellent", "perfect", "amazing",
           "beautiful", "nice", "comfortable", "happy", "quality"}
    NEG = {"bad", "poor", "cheap", "worst", "horrible", "terrible",
           "disappointed", "fake", "damaged", "wrong", "late", "faded"}

    def predict_heuristic(text: str) -> dict:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        pos   = len(words & POS)
        neg   = len(words & NEG)
        if pos >= neg:
            return {"label": "POSITIVE",
                    "score": round(min(0.55 + pos * 0.04, 0.99), 3),
                    "model": "Keyword heuristic"}
        return {"label": "NEGATIVE",
                "score": round(min(0.55 + neg * 0.04, 0.99), 3),
                "model": "Keyword heuristic"}

    return predict_heuristic, "Keyword heuristic (minimal fallback)"


# ═══════════════════════════════════════════════════════════════════
# 4.  ANALYTICS HELPERS
# ═══════════════════════════════════════════════════════════════════

TIER_ORDER = ["Budget", "Value", "Mid-Range", "Premium", "Luxury"]
POP_ORDER  = ["Niche (<100)", "Growing (100-999)", "Popular (1K-5K)",
              "Trending (5K-10K)", "Viral (10K+)"]


def most_complained_feature(df: pd.DataFrame) -> str:
    neg_text = " ".join(
        df.loc[df["BERT_Label"] == "NEGATIVE", "review_text"]
          .fillna("").str.lower().tolist()
    )
    if not neg_text.strip():
        return "N/A"
    scores = {
        feat: sum(neg_text.count(kw) for kw in kws)
        for feat, kws in FEATURE_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "N/A"


def top_negative_words(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    neg_reviews = df.loc[df["BERT_Label"] == "NEGATIVE", "review_text"].fillna("")
    words    = re.findall(r"\b[a-z]{3,}\b", " ".join(neg_reviews).lower())
    filtered = [w for w in words if w not in STOP_WORDS]
    counts   = Counter(filtered).most_common(top_n)
    return pd.DataFrame(counts, columns=["word", "count"])


def build_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    present_tiers = [t for t in TIER_ORDER if t in df["Price_Tier"].unique()]
    present_pops  = [p for p in POP_ORDER  if p in df["Popularity_Bin"].unique()]
    pivot = (
        df.groupby(["Price_Tier", "Popularity_Bin"])["BERT_Score"]
          .mean()
          .unstack(fill_value=np.nan)
          .reindex(index=present_tiers, columns=present_pops)
    )
    return pivot


# ═══════════════════════════════════════════════════════════════════
# 5.  PLOT BUILDERS
# ═══════════════════════════════════════════════════════════════════

def _base_layout(**kwargs) -> dict:
    """Merge PLOTLY_TEMPLATE layout with extra kwargs."""
    base = PLOTLY_TEMPLATE["layout"].to_plotly_json()
    base.update(kwargs)
    return base


def plot_sentiment_heatmap(pivot: pd.DataFrame) -> go.Figure:
    z_vals = pivot.values
    annots = [[f"{v:.2f}" if not np.isnan(v) else "–" for v in row]
               for row in z_vals]

    fig = go.Figure(go.Heatmap(
        z            = z_vals,
        x            = list(pivot.columns),
        y            = list(pivot.index),
        colorscale   = [[0, PALETTE["negative"]],
                        [0.5, "#fbbf24"],
                        [1, PALETTE["positive"]]],
        zmin=0.4, zmax=1.0,
        text=annots, texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> x <b>%{x}</b><br>Avg Score: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Avg Score", tickformat=".2f", thickness=14, len=0.85),
    ))
    fig.update_layout(**_base_layout(
        title  = dict(text="Avg Sentiment Score · Price x Popularity",
                      font=dict(size=13), x=0),
        xaxis  = dict(title="Popularity Bin", tickangle=-25,
                      gridcolor=PALETTE["border"]),
        yaxis  = dict(title="Price Tier", gridcolor=PALETTE["border"]),
        height = 360,
    ))
    return fig


def plot_negative_words(word_df: pd.DataFrame) -> go.Figure:
    if word_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No negative reviews in current selection",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False,
                           font=dict(color=PALETTE["text_muted"], size=13))
        fig.update_layout(**_base_layout(height=360))
        return fig

    word_df = word_df.sort_values("count")
    fig = go.Figure(go.Bar(
        x=word_df["count"], y=word_df["word"], orientation="h",
        marker=dict(
            color=word_df["count"],
            colorscale=[[0, PALETTE["surface2"]], [1, PALETTE["negative"]]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Mentions: %{x}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Top Words in Negative Reviews", font=dict(size=13), x=0),
        xaxis=dict(title="Mentions", gridcolor=PALETTE["border"]),
        yaxis=dict(title="", tickfont=dict(size=11)),
        height=360,
    ))
    return fig


def plot_confidence_gauge(score: float, label: str) -> go.Figure:
    colour = PALETTE["positive"] if label == "POSITIVE" else PALETTE["negative"]
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(score * 100, 1),
        number= dict(suffix="%", font=dict(size=32, color=colour)),
        gauge = dict(
            axis      = dict(range=[50, 100], tickwidth=1,
                             tickcolor=PALETTE["border"],
                             tickfont=dict(color=PALETTE["text_muted"])),
            bar       = dict(color=colour, thickness=0.28),
            bgcolor   = PALETTE["surface2"],
            borderwidth=0,
            steps=[
                dict(range=[50,  65], color=PALETTE["surface"]),
                dict(range=[65,  80], color="#1e293b"),
                dict(range=[80, 100], color="#0f172a"),
            ],
            threshold=dict(line=dict(color=colour, width=3),
                           value=round(score * 100, 1)),
        ),
    ))
    fig.update_layout(**_base_layout(height=220, margin=dict(l=20, r=20, t=10, b=10)))
    return fig


# ═══════════════════════════════════════════════════════════════════
# 6.  SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:12px 0 18px'>"
            f"<span style='font-size:1.6rem'>✦</span>"
            f"<div style='font-size:0.95rem;font-weight:700;"
            f"color:{PALETTE['text']};margin-top:6px'>Fashion-Tech</div>"
            f"<div style='font-size:0.7rem;color:{PALETTE['text_muted']};"
            f"letter-spacing:0.08em'>RETAIL INTELLIGENCE</div></div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Price Tier
        all_tiers = [t for t in TIER_ORDER if t in df["Price_Tier"].unique()]
        st.markdown("Price Tier")
        sel_tiers = st.multiselect(
            "price_tier_ms", options=all_tiers, default=all_tiers,
            label_visibility="collapsed",
        )
        st.markdown(" ")

        # Popularity Bin
        all_pops = [p for p in POP_ORDER if p in df["Popularity_Bin"].unique()]
        st.markdown("Popularity Bin")
        sel_pops = st.multiselect(
            "pop_bin_ms", options=all_pops, default=all_pops,
            label_visibility="collapsed",
        )
        st.markdown(" ")

        # Age slider (if column exists with variance)
        if "Age" in df.columns and df["Age"].nunique() > 5:
            age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
            st.markdown("Customer Age Range")
            age_range = st.slider(
                "age_slider", min_value=age_min, max_value=age_max,
                value=(age_min, age_max), label_visibility="collapsed",
            )
            age_mask = df["Age"].between(*age_range)
        else:
            age_mask = pd.Series(True, index=df.index)

        # Rating slider
        r_min = max(1.0, float(df["rating"].min()))
        r_max = min(5.0, float(df["rating"].max()))
        st.markdown(" ")
        st.markdown("Rating Range")
        rating_range = st.slider(
            "rating_slider", min_value=r_min, max_value=r_max,
            value=(r_min, r_max), step=0.5,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(
            f"<div style='font-size:0.68rem;color:{PALETTE['text_muted']};"
            f"line-height:1.6'>Stage 3 · Streamlit Dashboard<br>"
            f"DistilBERT SST-2 &amp; Plotly Express</div>",
            unsafe_allow_html=True,
        )

    mask = (
        df["Price_Tier"].isin(sel_tiers or all_tiers)
        & df["Popularity_Bin"].isin(sel_pops or all_pops)
        & df["rating"].between(*rating_range)
        & age_mask
    )
    return df[mask].copy()


# ═══════════════════════════════════════════════════════════════════
# 7.  DASHBOARD ROWS
# ═══════════════════════════════════════════════════════════════════

def render_header(is_mock: bool) -> None:
    st.markdown(
        f"<h1 style='font-size:1.9rem;font-weight:800;letter-spacing:-0.02em;"
        f"color:{PALETTE['text']};margin-bottom:2px'>✦ Retail Intelligence Dashboard</h1>"
        f"<p style='color:{PALETTE['text_muted']};font-size:0.85rem;"
        f"margin-top:0;margin-bottom:18px'>Fashion-Tech · Myntra Products · Stage 3</p>",
        unsafe_allow_html=True,
    )
    if is_mock:
        st.info(
            "**Demo mode** — `processed_fashion_reviews.csv` not found. "
            "Running on synthetic data. Place your Stage 2 CSV alongside "
            "`app.py` and relaunch to use real data.",
            icon="ℹ️",
        )


def render_kpi_row(df: pd.DataFrame) -> None:
    st.markdown(
        "<div class='section-header'>Key Performance Indicators</div>",
        unsafe_allow_html=True,
    )
    total     = len(df)
    avg_score = df["BERT_Score"].mean() if total else 0.0
    complaint = most_complained_feature(df)
    pct_neg   = (df["BERT_Label"] == "NEGATIVE").mean() * 100 if total else 0.0

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Total Filtered Products</div>"
            f"<div class='kpi-value'>{total:,}</div>"
            f"<div class='kpi-sub'>{pct_neg:.1f}% flagged negative</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        sc = (PALETTE["positive"] if avg_score >= 0.65
              else PALETTE["negative"] if avg_score < 0.55 else "#fbbf24")
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Avg Sentiment Score</div>"
            f"<div class='kpi-value' style='color:{sc}'>{avg_score:.3f}</div>"
            f"<div class='kpi-sub'>DistilBERT confidence (0.5 – 1.0)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>Most Complained Feature</div>"
            f"<div class='kpi-value' style='font-size:1.2rem;"
            f"color:{PALETTE[\"negative\"]}'>{complaint}</div>"
            f"<div class='kpi-sub'>from negative review text</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)


def render_analytics_hub(df: pd.DataFrame) -> None:
    st.markdown(
        "<div class='section-header'>Analytics Hub</div>",
        unsafe_allow_html=True,
    )
    col_l, col_r = st.columns([1.15, 1], gap="large")

    with col_l:
        if df.empty:
            st.warning("No data matches current filters — try widening the sidebar controls.")
        else:
            pivot = build_heatmap_data(df)
            if pivot.empty or pivot.isnull().all().all():
                st.info("Not enough cross-segment data for heatmap. Widen your filters.")
            else:
                st.plotly_chart(plot_sentiment_heatmap(pivot), use_container_width=True)

    with col_r:
        st.plotly_chart(plot_negative_words(top_negative_words(df)),
                        use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)


def render_sentiment_tester(predict_fn, model_name: str) -> None:
    st.markdown(
        "<div class='section-header'>Interactive Sentiment Tester</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Model in use: **{model_name}**")

    review_input = st.text_area(
        "Type or paste a product review:",
        placeholder=(
            'e.g. "The fabric feels cheap for a premium dress, '
            'but the colour is absolutely gorgeous."'
        ),
        height=110,
        key="review_input",
    )

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        analyse = st.button("Analyse Sentiment →", type="primary",
                            use_container_width=True)

    if analyse:
        text = review_input.strip()
        if not text:
            st.warning("Please enter some review text first.")
            return

        with st.spinner("Running inference …"):
            result = predict_fn(text)

        label = result["label"]
        score = result["score"]
        emoji = "✅" if label == "POSITIVE" else "⚠️"
        css   = "positive-label" if label == "POSITIVE" else "negative-label"

        res_col, gauge_col = st.columns([1.4, 1], gap="large")

        with res_col:
            st.markdown(
                f"<div class='result-box'>"
                f"<div class='kpi-label'>Prediction</div>"
                f"<div class='result-label {css}'>{emoji} {label}</div>"
                f"<div style='margin-top:12px;color:{PALETTE['text_muted']};"
                f"font-size:0.82rem;line-height:1.8'>"
                f"<b>Confidence</b> : {score*100:.1f}%<br>"
                f"<b>Model</b>      : {result['model']}<br>"
                f"<b>Word count</b> : {len(text.split())} words"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            bar_color = (PALETTE["positive"] if label == "POSITIVE"
                         else PALETTE["negative"])
            pct = int((score - 0.5) / 0.5 * 100)   # normalise [0.5,1] -> [0,100]
            st.markdown(
                f"<div style='font-size:0.75rem;color:{PALETTE['text_muted']};"
                f"margin-bottom:5px'>Confidence "
                f"<span style='color:{bar_color};font-weight:700'>"
                f"{score*100:.1f}%</span></div>"
                f"<div style='background:{PALETTE[\"surface2\"]};border-radius:99px;"
                f"height:10px;overflow:hidden'>"
                f"<div style='background:{bar_color};width:{pct}%;"
                f"height:100%;border-radius:99px'></div></div>",
                unsafe_allow_html=True,
            )

        with gauge_col:
            st.plotly_chart(plot_confidence_gauge(score, label),
                            use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# 8.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    with st.spinner("Loading dataset …"):
        df, is_mock = load_data("processed_fashion_reviews.csv")

    df_filtered = render_sidebar(df)
    render_header(is_mock)
    render_kpi_row(df_filtered)
    st.divider()
    render_analytics_hub(df_filtered)
    st.divider()

    with st.spinner("Initialising sentiment model (first load only) …"):
        predict_fn, model_name = load_sentiment_model()

    render_sentiment_tester(predict_fn, model_name)


if __name__ == "__main__":
    main()
