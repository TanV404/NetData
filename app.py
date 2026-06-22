import streamlit as st
import pandas as pd
from src.sentiment_analysis import add_sentiment, HAS_TRANSFORMER
from src.recommender import build_recommender, get_hybrid_similarity, recommend, evaluate_recommender

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="NetData",
    layout="wide",
)

st.title("🎬 NetData")
st.caption("Explore, analyze, and get recommendations using data-driven insights.")

# --- Load and Process Data ---
@st.cache_data
def load_data():
    try:
        # Load local data file
        df = pd.read_csv("data/netflix_titles.csv", quotechar='"', on_bad_lines='skip', encoding='utf-8')
    except FileNotFoundError:
        # Fallback to github remote URL
        url = "https://raw.githubusercontent.com/TanV404/Netflix-Dashboard/main/data/netflix_titles.csv"
        df = pd.read_csv(url, quotechar='"', on_bad_lines='skip', encoding='utf-8')
    
    # Compute VADER sentiment for the dataset
    df = add_sentiment(df)
    return df

@st.cache_resource
def load_recommender_matrices(df):
    return build_recommender(df)

df = load_data()
df_clean, sim_desc, sim_genre, sim_rating, indices = load_recommender_matrices(df)

# --- Tabs Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💬 Sentiment Analysis", "🎯 Recommendations", "📈 Trend Analysis"])

# ======================
# Tab 1: Overview
# ======================
with tab1:
    st.subheader("📊 Dataset Overview")

    # Sidebar Filters (moved inside tab 1)
    st.write("🎛️ Filter Options")

    def split_unique_values(series):
        """Split comma-separated values and return sorted unique cleaned list."""
        all_items = []
        for val in series.dropna():
            all_items.extend([
                v.strip() for v in val.split(",")
                if v.strip() and v.strip().lower() != "unknown"
            ])
        return sorted(set(all_items))

    # Create clean unique lists
    type_options = sorted(df["type"].dropna().unique())
    country_options = split_unique_values(df["country"])
    genre_options = split_unique_values(df["listed_in"])

    # Streamlit filter widgets
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_type = st.multiselect("Select Type", options=type_options)
    with col2:
        selected_country = st.multiselect("Select Country", options=country_options)
    with col3:
        selected_genre = st.multiselect("Select Genre", options=genre_options)

    # Apply filters
    filtered_df = df.copy()
    if selected_type:
        filtered_df = filtered_df[filtered_df["type"].isin(selected_type)]
    if selected_country:
        filtered_df = filtered_df[
            filtered_df["country"].apply(
                lambda x: any(c in str(x) for c in selected_country)
            )
        ]
    if selected_genre:
        filtered_df = filtered_df[
            filtered_df["listed_in"].apply(
                lambda x: any(g in str(x) for g in selected_genre)
            )
        ]

    # --- Charts ---
    col1, col2 = st.columns(2)
    with col1:
        st.write("🎞️ Titles by Type")
        st.bar_chart(filtered_df["type"].value_counts())

    with col2:
        st.write("🌍 Titles by Country")
        country_counts = filtered_df["country"].value_counts().head(10)
        st.bar_chart(country_counts)

# ======================
# Tab 2: Sentiment Analysis
# ======================
with tab2:
    st.subheader("💬 Sentiment Analysis on Descriptions")
    st.caption("Dataset sentiment has been upgraded from TextBlob to VADER (Valence Aware Dictionary and sEntiment Reasoner).")

    # Use filtered data from Tab 1
    filtered_df["sentiment"] = filtered_df["sentiment"].astype(str).str.lower()

    sentiment_counts = filtered_df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("😀 Positive Samples")
        positive_samples = filtered_df[filtered_df["sentiment"] == "positive"][["title", "description"]].head(5)
        if len(positive_samples) == 0:
            st.info("No positive samples found.")
        else:
            st.dataframe(positive_samples, hide_index=True)

    with col2:
        st.subheader("😔 Negative Samples")
        negative_samples = filtered_df[filtered_df["sentiment"] == "negative"][["title", "description"]].tail(5)
        if len(negative_samples) == 0:
            st.info("No negative samples found.")
        else:
            st.dataframe(negative_samples, hide_index=True)

    # Sentiment Prediction Box
    st.subheader("🧠 Try Your Own Description")
    
    col_input, col_model = st.columns([3, 1])
    with col_input:
        user_input = st.text_input("Enter a movie/show description to analyze sentiment:")
    with col_model:
        if HAS_TRANSFORMER:
            model_options = ["VADER (Fast, Lexicon)", "DistilBERT (Transformer, High Accuracy)"]
            help_text = "Select the sentiment model."
        else:
            model_options = ["VADER (Fast, Lexicon)"]
            help_text = "DistilBERT (Transformer) is disabled to prevent cloud deployment OOM crashes. Install torch/transformers locally to enable."
        model_choice = st.selectbox("Select Model", options=model_options, help=help_text)

    if user_input.strip():
        if model_choice.startswith("VADER"):
            from src.sentiment_analysis import analyze_sentiment_vader
            sentiment, score = analyze_sentiment_vader(user_input)
            st.success(f"**Predicted Sentiment:** {sentiment}")
            st.markdown(fr"**VADER Compound Score:** `{score:.4f}` (Positive $\ge 0.05$, Negative $\le -0.05$, Neutral between)")
        elif HAS_TRANSFORMER and model_choice.startswith("DistilBERT"):
            from src.sentiment_analysis import analyze_sentiment_transformer
            with st.spinner("Analyzing with DistilBERT... (loading weights on first run)"):
                sentiment, score = analyze_sentiment_transformer(user_input)
            st.success(f"**Predicted Sentiment:** {sentiment}")
            st.markdown(f"**DistilBERT Confidence Score:** `{score:.4f}`")

# ======================
# Tab 3: Recommendations
# ======================
with tab3:
    st.subheader("🎯 Hybrid Content Recommender")
    st.caption("A hybrid recommender combining description semantics, genre overlaps, and rating compatibility.")

    # 1. Tuning Weights
    st.markdown("### 🔧 Tune Similarity Weights")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        w_desc = st.slider("Description Semantics (TF-IDF)", 0.0, 1.0, 0.4, 0.1, help="Weight for the similarity of show descriptions.")
    with col_w2:
        w_genre = st.slider("Genre Overlap", 0.0, 1.0, 0.4, 0.1, help="Weight for the similarity of genres (listed_in).")
    with col_w3:
        w_rating = st.slider("Rating Compatibility", 0.0, 1.0, 0.2, 0.1, help="Weight for content rating and maturity level matching.")

    # Compute hybrid similarity
    cosine_sim = get_hybrid_similarity(sim_desc, sim_genre, sim_rating, w_desc, w_genre, w_rating)

    # 2. Recommendation Engine Interface
    st.markdown("### 🔍 Search Recommendations")
    user_input = st.text_input("Enter a title to get similar recommendations:")
    if user_input:
        recs = recommend(user_input, df_clean, cosine_sim, indices)
        if recs:
            st.success(f"Top {len(recs)} recommendations for **{user_input.title()}**:")
            for r in recs:
                show = df_clean[df_clean["title"] == r.lower()].iloc[0]
                with st.expander(f"🎥 {r}"):
                    st.write(f"**Genre:** {show['listed_in']}")
                    st.write(f"**Rating:** {show['rating']}")
                    st.write(f"**Description:** {show['description']}")
        else:
            st.warning("❌ No similar titles found. Try another one.")
    else:
        st.info("💡 Type a show or movie title above to get recommendations!")

    # 3. Offline Evaluation Metrics
    st.markdown("---")
    st.subheader("📊 Recommender Offline Evaluation")
    st.caption("These metrics evaluate the current hybrid configuration across a random sample of 200 items.")

    # Cache evaluation results based on the similarity matrix to avoid slow re-renders
    @st.cache_data(ttl=600)
    def get_evaluation_results(df_clean, sim_matrix):
        return evaluate_recommender(df_clean, sim_matrix, k=5, sample_size=200)

    eval_results = get_evaluation_results(df_clean, cosine_sim)

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric(label="Average Precision @ 5 (Genre-based)", value=f"{eval_results['precision_at_k']:.2%}")
        st.caption("Fraction of recommendations sharing at least one genre with the query.")
    with col_e2:
        st.metric(label="Catalog Coverage @ 5", value=f"{eval_results['catalog_coverage']:.2%}")
        st.caption("Percentage of unique items recommended in top-5 across all catalog items.")
    with col_e3:
        st.metric(label="Recommendation Diversity @ 5", value=f"{eval_results['diversity']:.2%}")
        st.caption("Average Jaccard distance between genres of recommended items (closer to 1 = more diverse).")


# ======================
# Tab 4: Trend Analysis
# ======================
with tab4:
    st.subheader("📈 Content Trend Analysis")
    st.caption("Explore how Netflix's library composition and genre focus have shifted over time.")

    # Data preparation for trends
    df_trends = df.copy()
    df_trends['date_added'] = pd.to_datetime(df_trends['date_added'].str.strip(), errors='coerce')
    df_trends = df_trends.dropna(subset=['date_added'])
    df_trends['year_added'] = df_trends['date_added'].dt.year.astype(int)

    # 1. Content Added over Time (Movies vs TV Shows)
    st.markdown("### 📅 Content Added to Netflix per Year")
    year_counts = df_trends.groupby(['year_added', 'type']).size().unstack().fillna(0)
    year_counts = year_counts[year_counts.index >= 2008]
    year_counts.index = year_counts.index.astype(str)
    st.area_chart(year_counts)

    # 2. Genre Evolution
    st.markdown("### 🧬 Genre Popularity Trends (Titles Added per Year)")
    df_genre_trends = df_trends.copy()
    df_genre_trends['genre'] = df_genre_trends['listed_in'].apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
    df_genre_trends = df_genre_trends.explode('genre')
    top_genres = df_genre_trends['genre'].value_counts().head(5).index.tolist()
    
    df_genre_filtered = df_genre_trends[df_genre_trends['genre'].isin(top_genres)]
    genre_year_counts = df_genre_filtered.groupby(['year_added', 'genre']).size().unstack().fillna(0)
    genre_year_counts = genre_year_counts[genre_year_counts.index >= 2008]
    genre_year_counts.index = genre_year_counts.index.astype(str)
    st.line_chart(genre_year_counts)

    # 3. Country Trends
    st.markdown("### 🌍 Top Countries Content Production Growth")
    df_country_trends = df_trends.copy()
    df_country_trends['country_clean'] = df_country_trends['country'].apply(lambda x: [c.strip() for c in str(x).split(',') if c.strip() and c.strip().lower() != 'unknown'])
    df_country_trends = df_country_trends.explode('country_clean')
    top_countries = df_country_trends['country_clean'].value_counts().head(5).index.tolist()
    
    df_country_filtered = df_country_trends[df_country_trends['country_clean'].isin(top_countries)]
    country_year_counts = df_country_filtered.groupby(['year_added', 'country_clean']).size().unstack().fillna(0)
    country_year_counts = country_year_counts[country_year_counts.index >= 2008]
    country_year_counts.index = country_year_counts.index.astype(str)
    st.line_chart(country_year_counts)

# --- Footer ---
st.markdown("---")
st.caption("Netflix Dashboard • Built with ❤️ using Streamlit")
