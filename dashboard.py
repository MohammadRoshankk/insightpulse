"""
InsightPulse — Customer Review Intelligence Platform
Modern, high-fidelity Streamlit frontend.
"""

import os
import math
import html
import sys
from datetime import datetime
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv

# Internal database and processing modules
from database import SessionLocal, Review
from clean_text import clean_text
from detect_category import detect_category
from tfidf_keywords import get_top_tfidf_words
from keyword_extraction import get_bigrams

load_dotenv()

CLOUD_MODE = os.getenv("CLOUD_MODE") == "1"

def predict_sentiment(text):
    """Load the model only when it is actually needed (never in cloud mode)."""
    from transformer_sentiment import transformer_label
    return transformer_label(text)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="InsightPulse — Customer Review Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling System (Tokens, Cards, Badges, Typography)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background and overall spacing */
    .stApp {
        background-color: #F8FAFC;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Branded Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #7C3AED 100%);
        border-radius: 16px;
        padding: 24px 28px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
    }
    .hero-banner h1 {
        color: white !important;
        font-size: 26px;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .hero-banner p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 14.5px;
        margin: 0;
        font-weight: 400;
    }

    /* Section Headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin: 20px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* KPI Cards Grid */
    .kpi-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 6px -1px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -3px rgba(0, 0, 0, 0.06);
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .kpi-title {
        font-size: 12.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }
    .kpi-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #64748B;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 11.5px;
        font-weight: 600;
        text-transform: capitalize;
    }
    .badge-pos {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    .badge-neu {
        background-color: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
    }
    .badge-neg {
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }
    .badge-complaint {
        background-color: #FFE4E6;
        color: #E11D48;
        border: 1px solid #FDA4AF;
    }
    .badge-feature {
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
    }
    .badge-general {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }

    /* Review Card */
    .review-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .review-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .review-stars {
        color: #F59E0B;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .review-text {
        font-size: 13.5px;
        color: #1E293B;
        line-height: 1.5;
        margin: 8px 0;
    }
    .review-date {
        font-size: 11.5px;
        color: #94A3B8;
    }

    /* Keyword Tags */
    .keyword-tag {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 4px;
    }
    .keyword-weight {
        font-size: 10.5px;
        color: #6366F1;
        margin-left: 4px;
    }

    /* AI Executive Briefing Cards */
    .briefing-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
    .briefing-card h3 {
        margin-top: 0;
        font-size: 16px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Custom Streamlit adjustments */
    div[data-testid="stSidebarHeader"] {
        padding-top: 1rem;
    }
    .css-1544g2n {
        padding: 1rem 1rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Data Layer (Safe, Read-Only with Refresh Mechanism)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_reviews_data():
    """Queries SQLite via SQLAlchemy and returns a clean, preprocessed DataFrame."""
    session = SessionLocal()
    try:
        reviews = session.query(Review).all()
        if not reviews:
            return pd.DataFrame()
        
        data = [{
            "id": r.id,
            "text": r.text if r.text else "",
            "rating": float(r.rating) if r.rating is not None else 3.0,
            "date": r.date,
            "sentiment": str(r.sentiment).lower() if r.sentiment else "neutral",
            "category": str(r.category).lower() if r.category else "general"
        } for r in reviews]
        
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["month"] = df["date"].dt.to_period("M").astype(str)
        else:
            df["date"] = pd.Timestamp.now()
            df["month"] = df["date"].dt.to_period("M").astype(str)
            
        return df
    finally:
        session.close()


def render_star_glyphs(rating: float) -> str:
    """Formats numeric ratings into star glyphs."""
    try:
        r = int(round(float(rating)))
        r = max(1, min(5, r))
        return "★" * r + "☆" * (5 - r)
    except Exception:
        return "★★★☆☆"


# Load live dataset
df_raw = load_reviews_data()


# ---------------------------------------------------------
# Sidebar: Brand, Navigation, and Dataset Health Status
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
            <div style="background: linear-gradient(135deg, #4F46E5, #7C3AED); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 20px; font-weight: 800;">
                ⚡
            </div>
            <div>
                <div style="font-size: 18px; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">InsightPulse</div>
                <div style="font-size: 11px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">AI Intelligence</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        label="Navigation",
        options=[
            "📊 Dashboard Overview",
            "💬 Review Explorer",
            "📈 Sentiment & Categories",
            "🧠 Topics & Keywords",
            "✨ AI Executive Summary",
            "📥 Data Management",
            "🔌 API Access & Sandbox"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Data health stats in sidebar
    st.markdown("<div style='font-size: 12px; font-weight: 700; color: #64748B; margin-bottom: 8px;'>DATABASE STATUS</div>", unsafe_allow_html=True)
    if not df_raw.empty:
        demo_badge_html = "<div style='margin-bottom: 8px;'><span class='badge badge-neu'>Read-only demo</span></div>" if CLOUD_MODE else ""
        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px; font-size: 12px; color: #475569; margin-bottom: 12px;">
                {demo_badge_html}
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>Reviews Stored:</span>
                    <strong style="color: #0F172A;">{len(df_raw):,}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>Date Span:</span>
                    <strong style="color: #0F172A;">{df_raw['date'].dt.strftime('%b %Y').min()} – {df_raw['date'].dt.strftime('%b %Y').max()}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Model Engine:</span>
                    <span style="color: #059669; font-weight: 600;">DistilBERT SST-2 (pretrained)</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Database has no records loaded.")

    request_url = os.getenv("REQUEST_ACCESS_URL", "").strip()
    if request_url:
        st.link_button("Request API access", url=request_url, width="stretch")

    if st.button("🔄 Refresh Data Cache", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
        <div style="margin-top: 24px; font-size: 11px; color: #94A3B8; text-align: center;">
            InsightPulse Platform v1.2<br>FastAPI · SQLite · DistilBERT · Gemini
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Empty State Guard
# ---------------------------------------------------------
if df_raw.empty:
    st.info("👋 Welcome to InsightPulse! The database has no reviews yet.")
    st.markdown("Run `python3 load_data_to_db.py`, then `python3 update_analysis_in_db.py` locally to populate it.")
    st.stop()


# =========================================================
# VIEW 1: 📊 DASHBOARD OVERVIEW (Primary Executive Screen)
# =========================================================
if nav_selection == "📊 Dashboard Overview":

    # Hero Banner matching reference design
    st.markdown("""
        <div class="hero-banner">
            <h1>Welcome to InsightPulse 👋</h1>
            <p>Explore customer feedback, discover emerging trends, and surface actionable AI-powered business insights.</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. KPI Scorecard Row (4 Columns)
    total_reviews = len(df_raw)
    avg_rating = round(df_raw["rating"].mean(), 2)

    sent_counts = df_raw["sentiment"].value_counts()
    pos_count = sent_counts.get("positive", 0)
    neu_count = sent_counts.get("neutral", 0)
    neg_count = sent_counts.get("negative", 0)

    pos_pct = round((pos_count / total_reviews) * 100, 1) if total_reviews > 0 else 0
    neu_pct = round((neu_count / total_reviews) * 100, 1) if total_reviews > 0 else 0
    neg_pct = round((neg_count / total_reviews) * 100, 1) if total_reviews > 0 else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-header">
                    <span class="kpi-title">Total Reviews</span>
                    <div class="kpi-icon" style="background: #EEF2FF; color: #4F46E5;">💬</div>
                </div>
                <div class="kpi-value">{total_reviews:,}</div>
                <div class="kpi-subtext"><span style="color: #059669; font-weight: 700;">● Active</span> Full dataset</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-header">
                    <span class="kpi-title">Positive</span>
                    <div class="kpi-icon" style="background: #ECFDF5; color: #10B981;">😊</div>
                </div>
                <div class="kpi-value">{pos_pct}%</div>
                <div class="kpi-subtext"><span class="badge badge-pos">{pos_count} reviews</span></div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-header">
                    <span class="kpi-title">Negative</span>
                    <div class="kpi-icon" style="background: #FEF2F2; color: #EF4444;">😡</div>
                </div>
                <div class="kpi-value">{neg_pct}%</div>
                <div class="kpi-subtext"><span class="badge badge-neg">{neg_count} reviews</span></div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-header">
                    <span class="kpi-title">Avg Rating</span>
                    <div class="kpi-icon" style="background: #FAF5FF; color: #7C3AED;">⭐</div>
                </div>
                <div class="kpi-value">{avg_rating} <span style="font-size: 16px; font-weight: 500; color: #94A3B8;">/ 5</span></div>
                <div class="kpi-subtext" style="color: #F59E0B; font-weight: 700;">{render_star_glyphs(avg_rating)}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 2. Main Analytics Charts Row (3 Columns)
    c1, c2, c3 = st.columns([1, 1, 1.2])

    with c1:
        st.markdown("<div class='section-header'>Sentiment Distribution</div>", unsafe_allow_html=True)
        sentiment_df = pd.DataFrame({
            "sentiment": ["positive", "neutral", "negative"],
            "count": [pos_count, neu_count, neg_count],
            "percentage": [f"{pos_pct}%", f"{neu_pct}%", f"{neg_pct}%"]
        })
        sentiment_df = sentiment_df[sentiment_df["count"] > 0]

        donut_chart = alt.Chart(sentiment_df).mark_arc(innerRadius=65, stroke="#ffffff", strokeWidth=2).encode(
            theta=alt.Theta(field="count", type="quantitative"),
            color=alt.Color(
                field="sentiment",
                type="nominal",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["#10B981", "#F59E0B", "#EF4444"]
                ),
                legend=alt.Legend(title=None, orient="bottom", labelFont="Plus Jakarta Sans", labelFontSize=12)
            ),
            tooltip=[
                alt.Tooltip("sentiment:N", title="Sentiment"),
                alt.Tooltip("count:Q", title="Total Count"),
                alt.Tooltip("percentage:N", title="Share")
            ]
        ).properties(height=260)
        st.altair_chart(donut_chart, width="stretch")

    with c2:
        st.markdown("<div class='section-header'>Ratings Distribution</div>", unsafe_allow_html=True)
        rating_counts = df_raw["rating"].round().astype(int).value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).reset_index()
        rating_counts.columns = ["rating", "count"]

        rating_bar = alt.Chart(rating_counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("rating:O", title="Star Rating", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title="Reviews"),
            color=alt.Color(
                "rating:O",
                scale=alt.Scale(
                    domain=[1, 2, 3, 4, 5],
                    range=["#EF4444", "#F97316", "#F59E0B", "#3B82F6", "#10B981"]
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip("rating:O", title="Rating"),
                alt.Tooltip("count:Q", title="Count")
            ]
        ).properties(height=260)
        st.altair_chart(rating_bar, width="stretch")

    with c3:
        st.markdown("<div class='section-header'>Review Trends Over Time</div>", unsafe_allow_html=True)
        monthly_trend = df_raw.groupby("month").agg(
            review_count=("id", "count"),
            avg_rating=("rating", "mean")
        ).reset_index()
        monthly_trend = monthly_trend.sort_values("month")
        monthly_trend["avg_rating"] = monthly_trend["avg_rating"].round(2)

        trend_line = alt.Chart(monthly_trend).mark_area(
            line={"color": "#4F46E5", "strokeWidth": 2.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(79, 70, 229, 0.35)", offset=0),
                    alt.GradientStop(color="rgba(79, 70, 229, 0.0)", offset=1)
                ],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X("month:N", title="Month", axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("review_count:Q", title="Volume"),
            tooltip=[
                alt.Tooltip("month:N", title="Month"),
                alt.Tooltip("review_count:Q", title="Reviews"),
                alt.Tooltip("avg_rating:Q", title="Avg Rating")
            ]
        ).properties(height=260)
        st.altair_chart(trend_line, width="stretch")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 3. Bottom Row: Categories, Keywords, and Recent Reviews
    b1, b2, b3 = st.columns([1, 1, 1.2])

    with b1:
        st.markdown("<div class='section-header'>Top Categories</div>", unsafe_allow_html=True)
        cat_counts = df_raw["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        cat_counts["pct"] = (cat_counts["count"] / total_reviews * 100).round(1)

        cat_chart = alt.Chart(cat_counts).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
            y=alt.Y("category:N", sort="-x", title=None),
            x=alt.X("count:Q", title="Reviews"),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(
                    domain=["complaint", "feature request", "general"],
                    range=["#E11D48", "#2563EB", "#64748B"]
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("pct:Q", title="Share %")
            ]
        ).properties(height=240)
        st.altair_chart(cat_chart, width="stretch")

    with b2:
        st.markdown("<div class='section-header'>Top Keywords</div>", unsafe_allow_html=True)
        try:
            cleaned_docs = [clean_text(t) for t in df_raw["text"] if t]
            tfidf_words = get_top_tfidf_words(cleaned_docs, top_n=12)
            tags_html = "<div style='display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 0;'>"
            for word, score in tfidf_words:
                tags_html += f"<span class='keyword-tag'>{word} <span class='keyword-weight'>{score:.2f}</span></span>"
            tags_html += "</div>"
            st.markdown(tags_html, unsafe_allow_html=True)
        except Exception as e:
            st.caption("Keyword extraction unavailable.")

    with b3:
        st.markdown("<div class='section-header'>Recent Customer Reviews</div>", unsafe_allow_html=True)
        recent_subset = df_raw.sort_values("date", ascending=False).head(3)
        for _, r in recent_subset.iterrows():
            s_class = "badge-pos" if r["sentiment"] == "positive" else ("badge-neg" if r["sentiment"] == "negative" else "badge-neu")
            c_class = "badge-complaint" if r["category"] == "complaint" else ("badge-feature" if r["category"] == "feature request" else "badge-general")
            date_str = r["date"].strftime("%b %d, %Y") if pd.notnull(r["date"]) else "Recent"
            text_snippet = html.escape((r["text"][:100] + "...") if len(r["text"]) > 100 else r["text"])

            st.markdown(f"""
                <div class="review-card">
                    <div class="review-card-header">
                        <span class="review-stars">{render_star_glyphs(r['rating'])}</span>
                        <div style="display: flex; gap: 6px;">
                            <span class="badge {s_class}">{html.escape(str(r['sentiment']))}</span>
                            <span class="badge {c_class}">{html.escape(str(r['category']))}</span>
                        </div>
                    </div>
                    <div class="review-text">"{text_snippet}"</div>
                    <div class="review-date">{date_str}</div>
                </div>
            """, unsafe_allow_html=True)


# =========================================================
# VIEW 2: 💬 REVIEW EXPLORER (Search, Filters, Star Ratings)
# =========================================================
elif nav_selection == "💬 Review Explorer":
    st.markdown("<div class='section-header' style='font-size: 24px;'>🔍 Customer Review Explorer</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Search, filter, and inspect individual customer reviews with full sentiment and category tagging.</p>", unsafe_allow_html=True)

    # Filter Toolbar
    with st.expander("Filter Controls", expanded=True):
        f1, f2, f3, f4 = st.columns([1.5, 1, 1, 1])
        with f1:
            search_query = st.text_input("Search reviews or keywords", placeholder="e.g. broke, shipping, quality...")
        with f2:
            sent_filter = st.multiselect("Sentiment", options=["positive", "neutral", "negative"], default=[])
        with f3:
            cat_filter = st.multiselect("Category", options=["complaint", "feature request", "general"], default=[])
        with f4:
            rating_filter = st.multiselect("Star Rating", options=[1, 2, 3, 4, 5], default=[])

    # Apply Filters
    filtered_df = df_raw.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df["text"].str.contains(search_query, case=False, na=False)]
    if sent_filter:
        filtered_df = filtered_df[filtered_df["sentiment"].isin(sent_filter)]
    if cat_filter:
        filtered_df = filtered_df[filtered_df["category"].isin(cat_filter)]
    if rating_filter:
        filtered_df = filtered_df[filtered_df["rating"].round().astype(int).isin(rating_filter)]

    # Result Header & Export
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"**Showing {len(filtered_df):,} of {len(df_raw):,} reviews**")
    with h2:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export CSV", data=csv_data, file_name="insightpulse_filtered_reviews.csv", mime="text/csv", width="stretch")

    # View Mode Toggle
    view_mode = st.radio("Display Format", ["Card View", "Table View"], horizontal=True, label_visibility="collapsed")

    if filtered_df.empty:
        st.info("No reviews match your selected filters. Try clearing your search or filter options.")
    elif view_mode == "Table View":
        display_table = filtered_df[["date", "rating", "sentiment", "category", "text"]].copy()
        display_table["date"] = display_table["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_table, width="stretch", height=450)
    else:
        # Paginated Card View
        PAGE_SIZE = 10
        total_pages = math.ceil(len(filtered_df) / PAGE_SIZE)
        page_num = st.number_input("Page", min_value=1, max_value=max(1, total_pages), step=1, value=1)
        start_idx = (page_num - 1) * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE

        for _, r in filtered_df.iloc[start_idx:end_idx].iterrows():
            s_class = "badge-pos" if r["sentiment"] == "positive" else ("badge-neg" if r["sentiment"] == "negative" else "badge-neu")
            c_class = "badge-complaint" if r["category"] == "complaint" else ("badge-feature" if r["category"] == "feature request" else "badge-general")
            date_str = r["date"].strftime("%B %d, %Y") if pd.notnull(r["date"]) else "Undated"

            escaped_text = html.escape(str(r['text']))

            st.markdown(f"""
                <div class="review-card">
                    <div class="review-card-header">
                        <span class="review-stars">{render_star_glyphs(r['rating'])} ({r['rating']:.1f})</span>
                        <div style="display: flex; gap: 8px;">
                            <span class="badge {s_class}">{html.escape(str(r['sentiment']))}</span>
                            <span class="badge {c_class}">{html.escape(str(r['category']))}</span>
                        </div>
                    </div>
                    <div class="review-text">"{escaped_text}"</div>
                    <div class="review-date">Submitted on {date_str} · Review ID #{r['id']}</div>
                </div>
            """, unsafe_allow_html=True)


# =========================================================
# VIEW 3: 📈 SENTIMENT & CATEGORIES (Deep-Dive Triage)
# =========================================================
elif nav_selection == "📈 Sentiment & Categories":
    st.markdown("<div class='section-header' style='font-size: 24px;'>📈 Sentiment & Category Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Detailed triage correlating sentiment predictions with star ratings and complaint detection.</p>", unsafe_allow_html=True)

    t1, t2 = st.columns(2)

    with t1:
        st.markdown("<div class='section-header'>Rating vs. Sentiment Alignment</div>", unsafe_allow_html=True)
        cross_df = df_raw.groupby(["rating", "sentiment"]).size().reset_index(name="count")
        heat_chart = alt.Chart(cross_df).mark_rect().encode(
            x=alt.X("rating:O", title="Star Rating"),
            y=alt.Y("sentiment:N", title="Predicted Sentiment"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="bluepurple"), title="Review Count"),
            tooltip=["rating", "sentiment", "count"]
        ).properties(height=280)
        st.altair_chart(heat_chart, width="stretch")

    with t2:
        st.markdown("<div class='section-header'>Sentiment Breakdown by Category</div>", unsafe_allow_html=True)
        cat_sent_df = df_raw.groupby(["category", "sentiment"]).size().reset_index(name="count")
        stacked_bar = alt.Chart(cat_sent_df).mark_bar().encode(
            x=alt.X("count:Q", title="Number of Reviews", stack="normalize"),
            y=alt.Y("category:N", title=None),
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(domain=["positive", "neutral", "negative"], range=["#10B981", "#F59E0B", "#EF4444"])
            ),
            tooltip=["category", "sentiment", "count"]
        ).properties(height=280)
        st.altair_chart(stacked_bar, width="stretch")

    # Actionable Complaint Triage Section
    st.markdown("<div class='section-header' style='color: #E11D48;'>⚠️ Actionable Complaints Triage</div>", unsafe_allow_html=True)
    complaints = df_raw[df_raw["category"] == "complaint"]
    critical_complaints = complaints[complaints["rating"] <= 2.0]

    st.markdown(f"Found **{len(complaints)} total complaints**, of which <span style='color: #DC2626; font-weight: 700;'>{len(critical_complaints)} are critical (1-2 stars)</span> requiring urgent attention.", unsafe_allow_html=True)

    for _, r in critical_complaints.head(8).iterrows():
        escaped_complaint = html.escape(str(r['text']))
        st.markdown(f"""
            <div class="review-card" style="border-left: 4px solid #EF4444;">
                <div class="review-card-header">
                    <span class="review-stars">{render_star_glyphs(r['rating'])}</span>
                    <span class="badge badge-neg">Critical Complaint</span>
                </div>
                <div class="review-text" style="font-weight: 500;">"{escaped_complaint}"</div>
                <div class="review-date">{r['date'].strftime('%b %d, %Y') if pd.notnull(r['date']) else 'Recent'}</div>
            </div>
        """, unsafe_allow_html=True)


# =========================================================
# VIEW 4: 🧠 TOPICS & KEYWORDS (TF-IDF & Bigram Phrases)
# =========================================================
elif nav_selection == "🧠 Topics & Keywords":
    st.markdown("<div class='section-header' style='font-size: 24px;'>🧠 Topics & Keyword Extraction</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Surface key themes, high-weight TF-IDF vocabulary, and recurring multi-word phrases.</p>", unsafe_allow_html=True)

    k_col1, k_col2 = st.columns(2)

    with k_col1:
        st.markdown("<div class='section-header'>Top Distinctive Words (TF-IDF)</div>", unsafe_allow_html=True)
        cleaned_docs = [clean_text(t) for t in df_raw["text"] if t]
        top_tfidf = get_top_tfidf_words(cleaned_docs, top_n=15)
        tfidf_df = pd.DataFrame(top_tfidf, columns=["word", "score"])

        tfidf_chart = alt.Chart(tfidf_df).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, color="#6366F1").encode(
            y=alt.Y("word:N", sort="-x", title=None),
            x=alt.X("score:Q", title="Mean TF-IDF Importance"),
            tooltip=["word", alt.Tooltip("score:Q", format=".3f")]
        ).properties(height=420)
        st.altair_chart(tfidf_chart, width="stretch")

    with k_col2:
        st.markdown("<div class='section-header'>Frequent Customer Phrases (Bigrams)</div>", unsafe_allow_html=True)
        all_bigrams = []
        for text in cleaned_docs:
            all_bigrams.extend(get_bigrams(text))

        from collections import Counter
        bigram_counts = Counter(all_bigrams).most_common(15)
        bigram_df = pd.DataFrame(bigram_counts, columns=["phrase", "occurrences"])

        bigram_chart = alt.Chart(bigram_df).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, color="#06B6D4").encode(
            y=alt.Y("phrase:N", sort="-x", title=None),
            x=alt.X("occurrences:Q", title="Frequency"),
            tooltip=["phrase", "occurrences"]
        ).properties(height=420)
        st.altair_chart(bigram_chart, width="stretch")


# =========================================================
# VIEW 5: ✨ AI EXECUTIVE SUMMARY (Gemini 3.6 Flash)
# =========================================================
elif nav_selection == "✨ AI Executive Summary":
    st.markdown("<div class='section-header' style='font-size: 24px;'>✨ AI-Generated Executive Summary</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Synthesize hundreds of customer reviews into a concise, board-ready executive memo using Gemini 3.6 Flash.</p>", unsafe_allow_html=True)

    summary_file_path = "data/executive_summary.md"

    if CLOUD_MODE:
        if os.path.exists(summary_file_path):
            with open(summary_file_path, "r", encoding="utf-8") as f:
                saved_summary = f.read()
            escaped_memo = html.escape(saved_summary)
            st.markdown(f"""
                <div class="briefing-card" style="border-left: 5px solid #7C3AED;">
                    <h3 style="color: #7C3AED;">Executive Memo</h3>
                    <div style="font-size: 15px; color: #1E293B; line-height: 1.6; white-space: pre-wrap;">
                        {escaped_memo}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No saved summary yet.")
    else:
        api_key_configured = bool(os.getenv("GOOGLE_API_KEY"))

        if not api_key_configured:
            st.warning("⚠️ `GOOGLE_API_KEY` is not detected in your `.env` file. Please set it to enable live Gemini generation.")
        else:
            st.markdown("""
                <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 12px; margin-bottom: 16px; font-size: 13px; color: #166534; display: flex; align-items: center; gap: 8px;">
                    <span>🟢</span>
                    <span><strong>Gemini 3.6 Flash Engine Active</strong> · Ready to synthesize reviews in batches of 50.</span>
                </div>
            """, unsafe_allow_html=True)

        if st.button("✨ Generate AI Executive Summary", type="primary"):
            with st.spinner("Analyzing and synthesizing reviews with Gemini 3.6 Flash..."):
                try:
                    from executive_summary import batch_summarize, final_summary
                    all_reviews = [clean_text(t) for t in df_raw["text"] if t]
                    batch_res = batch_summarize(all_reviews, batch_size=50)
                    synthesized = final_summary(batch_res)
                    st.session_state["gemini_summary"] = synthesized

                    os.makedirs("data", exist_ok=True)
                    with open(summary_file_path, "w", encoding="utf-8") as f:
                        f.write(synthesized)

                    st.success("Executive summary successfully generated and saved to data/executive_summary.md!")
                except Exception as e:
                    st.error(f"Failed to generate summary: {e}")

        # Display result if present in session state or file
        current_summary = st.session_state.get("gemini_summary")
        if not current_summary and os.path.exists(summary_file_path):
            with open(summary_file_path, "r", encoding="utf-8") as f:
                current_summary = f.read()

        if current_summary:
            escaped_memo = html.escape(current_summary)
            st.markdown(f"""
                <div class="briefing-card" style="border-left: 5px solid #7C3AED;">
                    <h3 style="color: #7C3AED;">Executive Memo</h3>
                    <div style="font-size: 15px; color: #1E293B; line-height: 1.6; white-space: pre-wrap;">
                        {escaped_memo}
                    </div>
                </div>
            """, unsafe_allow_html=True)


# =========================================================
# VIEW 6: 📥 DATA MANAGEMENT (Upload, Ingest, Health)
# =========================================================
elif nav_selection == "📥 Data Management":
    st.markdown("<div class='section-header' style='font-size: 24px;'>📥 Data Ingestion & Management</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Manage your SQLite database, inspect data ingestion status, or upload new customer review batches.</p>", unsafe_allow_html=True)

    if CLOUD_MODE:
        st.info("Read-only public demo. Uploading and reloading data are disabled. Run InsightPulse locally to ingest new reviews.")
        st.markdown("<div class='section-header'>Database Ingestion Status</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="briefing-card">
                <div style="font-size: 14px; margin-bottom: 8px;"><strong>Database Target:</strong> <code>sqlite:///data/public_reviews.db</code></div>
                <div style="font-size: 14px; margin-bottom: 8px;"><strong>Total Reviews Stored:</strong> {len(df_raw):,} records</div>
                <div style="font-size: 14px; margin-bottom: 8px;"><strong>Sentiment Model:</strong> DistilBERT SST-2 (Hugging Face)</div>
                <div style="font-size: 14px;"><strong>Category Detection:</strong> Rule-based Lexical Classifier</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        d1, d2 = st.columns([1.2, 1])

        with d1:
            st.markdown("<div class='section-header'>Upload and Analyze New Reviews</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload CSV file (Must contain 'text', 'rating', 'date' columns)", type=["csv"])

            if uploaded_file is not None:
                new_df = pd.read_csv(uploaded_file)
                st.markdown("**File Preview:**")
                st.dataframe(new_df.head(4), width="stretch")

                required_cols = {"text", "rating"}
                if not required_cols.issubset(new_df.columns):
                    st.error("Uploaded CSV is missing mandatory columns ('text' and 'rating').")
                else:
                    if st.button("🚀 Ingest & Enrich Uploaded Reviews", type="primary"):
                        with st.spinner("Processing sentiment and committing to SQLite database..."):
                            session = SessionLocal()
                            new_df = new_df.dropna(subset=["text", "rating"])
                            
                            added_count = 0
                            for _, row in new_df.iterrows():
                                review_date = pd.to_datetime(row.get("date", datetime.utcnow()))
                                cleaned = clean_text(str(row["text"]))
                                sent = predict_sentiment(str(row["text"]))
                                cat = detect_category(cleaned)

                                rev = Review(
                                    text=str(row["text"]),
                                    rating=float(row["rating"]),
                                    date=review_date,
                                    sentiment=sent,
                                    category=cat
                                )
                                session.add(rev)
                                added_count += 1
                            
                            session.commit()
                            session.close()
                            st.cache_data.clear()
                            st.success(f"Successfully processed and ingested {added_count} new reviews!")
                            st.rerun()

        with d2:
            st.markdown("<div class='section-header'>Database Ingestion Status</div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="briefing-card">
                    <div style="font-size: 14px; margin-bottom: 8px;"><strong>Database Target:</strong> <code>{os.getenv('DATABASE_URL', 'sqlite:///insightpulse.db')}</code></div>
                    <div style="font-size: 14px; margin-bottom: 8px;"><strong>Total Reviews Stored:</strong> {len(df_raw):,} records</div>
                    <div style="font-size: 14px; margin-bottom: 8px;"><strong>Sentiment Model:</strong> DistilBERT SST-2 (Hugging Face)</div>
                    <div style="font-size: 14px;"><strong>Category Detection:</strong> Rule-based Lexical Classifier</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-header'>Maintenance & Reload</div>", unsafe_allow_html=True)
            st.warning("Need to reset your database to original state?")
            confirm_reload = st.checkbox("I confirm I want to reset and reload the database")
            if st.button("⚠️ Reload Default Dataset (`sample_reviews.csv`)"):
                if not confirm_reload:
                    st.error("Please confirm before reloading the database.")
                else:
                    with st.spinner("Reloading and re-analyzing default dataset..."):
                        import subprocess
                        session = SessionLocal()
                        session.query(Review).delete()
                        session.commit()
                        session.close()
                        subprocess.run([sys.executable, "load_data_to_db.py"])
                        subprocess.run([sys.executable, "update_analysis_in_db.py"])
                        st.cache_data.clear()
                        st.success("Default dataset reloaded and enriched successfully!")
                        st.rerun()


# =========================================================
# VIEW 7: 🔌 API ACCESS & LIVE TESTER
# =========================================================
elif nav_selection == "🔌 API Access & Sandbox":
    st.markdown("<div class='section-header' style='font-size: 24px;'>🔌 API Access & Live Review Sandbox</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Integrate with the FastAPI backend or test model inference interactively on custom text.</p>", unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("<div class='section-header'>FastAPI Microservice Documentation</div>", unsafe_allow_html=True)
        st.markdown("""
            <div class="briefing-card">
                <p>The InsightPulse backend runs as a high-performance REST API built on <strong>FastAPI</strong> and containerized with Docker.</p>
                <div style="background: #F1F5F9; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 12.5px; margin: 12px 0;">
                    <strong>Endpoint:</strong> POST /analyze<br>
                    <strong>Header:</strong> x-api-key: [YOUR_CLIENT_KEY]<br>
                    <strong>Payload:</strong> {"text": "Great product, fast shipping!"}
                </div>
                <p style="font-size: 13px; color: #64748B;">Interactive Swagger documentation is available when running the API server at <code>http://localhost:8000/docs</code>.</p>
            </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("<div class='section-header'>Local Inference Sandbox</div>", unsafe_allow_html=True)
        st.caption("Note: This sandbox runs the model in-process and is not a call to the FastAPI service.")
        test_input = st.text_area("Enter a customer review to test live inference:", value="The product arrived damaged and customer support was unresponsive.", height=110)
        
        if CLOUD_MODE:
            st.info("Live model inference runs locally or via the Dockerized FastAPI service. It is disabled on the free cloud tier.")
            st.markdown("""
                **Example API request (cURL):**
                ```bash
                curl -X POST "http://localhost:8000/analyze" \\
                     -H "Content-Type: application/json" \\
                     -H "x-api-key: YOUR_CLIENT_KEY" \\
                     -d '{"text": "The product arrived damaged and customer support was unresponsive."}'
                ```
            """)
        else:
            if st.button("⚡ Test Live Inference", type="primary"):
                with st.spinner("Classifying sentiment and category..."):
                    cleaned = clean_text(test_input)
                    pred_sentiment = predict_sentiment(test_input)
                    pred_category = detect_category(cleaned)

                    st.markdown("**Model Output:**")
                    st.json({
                        "text": test_input,
                        "cleaned_text": cleaned,
                        "sentiment": pred_sentiment,
                        "category": pred_category
                    })