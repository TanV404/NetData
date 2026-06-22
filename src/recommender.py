"""
recommender.py
---------------
Hybrid content-based recommender system for the Netflix dataset
using description TF-IDF, genre matching, and content rating compatibility.
Also includes offline evaluation metrics (Precision@K, Catalog Coverage, Diversity).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from difflib import get_close_matches

try:
    import streamlit as st
except ImportError:
    # fallback if Streamlit isn't installed
    class st:
        @staticmethod
        def info(msg): print(msg)
        @staticmethod
        def error(msg): print(msg)


def build_recommender(df):
    """
    Build individual similarity matrices for description, genres, and ratings.

    Args:
        df (pd.DataFrame): Netflix dataset

    Returns:
        tuple: (cleaned dataframe, sim_desc, sim_genre, sim_rating, title indices)
    """
    df = df.copy()

    # Clean and normalize columns
    df['title'] = df['title'].astype(str).str.strip().str.lower()
    df['listed_in'] = df['listed_in'].fillna('')
    df['description'] = df['description'].fillna('')
    df['country'] = df.get('country', '').fillna('')
    df['director'] = df.get('director', '').fillna('')
    df['rating'] = df.get('rating', 'Unknown').fillna('Unknown')

    # 1. Description Similarity (TF-IDF on description only)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    sim_desc = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 2. Genre Similarity (MultiLabelBinarizer on listed_in)
    genres_list = df['listed_in'].apply(lambda x: [g.strip().lower() for g in x.split(',') if g.strip()])
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genres_list)
    sim_genre = cosine_similarity(genre_matrix, genre_matrix)

    # 3. Rating Similarity (Exact rating & maturity level)
    def get_rating_group(r):
        r = str(r).upper().strip()
        if r in ['G', 'TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G']:
            return 'kids'
        elif r in ['PG', 'PG-13', 'TV-PG', 'TV-14']:
            return 'teens'
        elif r in ['R', 'NC-17', 'TV-MA', 'UR', 'NR']:
            return 'adults'
        else:
            return 'unknown'

    df['rating_group'] = df['rating'].apply(get_rating_group)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    rating_features = ohe.fit_transform(df[['rating', 'rating_group']])
    sim_rating = cosine_similarity(rating_features, rating_features)

    # Create title → index mapping
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    return df, sim_desc, sim_genre, sim_rating, indices


def get_hybrid_similarity(sim_desc, sim_genre, sim_rating, w_desc, w_genre, w_rating):
    """
    Computes a weighted combination of similarity matrices.
    """
    total_w = w_desc + w_genre + w_rating
    if total_w == 0:
        return sim_desc  # Fallback

    w_d = w_desc / total_w
    w_g = w_genre / total_w
    w_r = w_rating / total_w

    return w_d * sim_desc + w_g * sim_genre + w_r * sim_rating


def recommend(title, df, cosine_sim, indices, n=3):
    """
    Recommend N similar titles based on hybrid similarity.
    """
    title = title.strip().lower()

    # Handle fuzzy title matching
    if title not in indices:
        close_matches = get_close_matches(title, indices.index, n=1, cutoff=0.6)
        if close_matches:
            suggestion = close_matches[0]
            st.info(f"🔍 Did you mean **{suggestion.title()}**?")
            title = suggestion
        else:
            st.error("❌ No such title found. Try another one.")
            return []

    # Get index of the given title
    idx = indices[title]
    
    # Handle duplicates safely
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    elif hasattr(idx, '__iter__'):
        idx = list(idx)[0]

    # Compute similarity scores for the title
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Filter out the item itself
    recs = [i for i, _ in sim_scores if i != idx][:n]

    # Get top N recommendations
    recommendations = df['title'].iloc[recs]

    # Capitalize for cleaner display
    return [t.title() for t in recommendations.tolist()]


def evaluate_recommender(df, sim_matrix, k=5, sample_size=200):
    """
    Computes offline evaluation metrics for the recommender:
    - Average Precision@K (based on genre match)
    - Catalog Coverage@K (percentage of unique items recommended)
    - Diversity@K (average Jaccard distance of genres among recommended items)
    """
    # 1. Prepare genre sets for comparison
    genres = df['listed_in'].apply(lambda x: set([g.strip().lower() for g in x.split(',') if g.strip()]))
    genres_dict = dict(zip(df.index, genres))

    # 2. Precision@K and Diversity@K (computed over a random sample for execution speed)
    np.random.seed(42)
    sample_indices = np.random.choice(df.index, size=min(sample_size, len(df)), replace=False)

    precisions = []
    diversities = []

    for idx in sample_indices:
        query_genres = genres_dict[idx]
        if not query_genres:
            continue

        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        recs = [i for i, _ in sim_scores if i != idx][:k]

        if not recs:
            continue

        # Precision@K: how many recommendations share at least one genre with the query?
        relevant_count = sum(1 for r in recs if query_genres.intersection(genres_dict[r]))
        precisions.append(relevant_count / len(recs))

        # Diversity@K: average Jaccard distance between all pairs of recommendations
        if len(recs) > 1:
            jaccard_distances = []
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    g_i = genres_dict[recs[i]]
                    g_j = genres_dict[recs[j]]
                    union = g_i.union(g_j)
                    intersection = g_i.intersection(g_j)
                    jaccard_sim = len(intersection) / len(union) if union else 0.0
                    jaccard_distances.append(1.0 - jaccard_sim)
            diversities.append(np.mean(jaccard_distances))
        else:
            diversities.append(0.0)

    # 3. Catalog Coverage@K: computed using vectorized argpartition for speed
    k_plus_1 = k + 1
    partitioned = np.argpartition(-sim_matrix, k_plus_1, axis=1)[:, :k_plus_1]

    all_recommended_indices = set()
    for i in range(len(df)):
        recs = partitioned[i]
        recs_filtered = recs[recs != i][:k]
        all_recommended_indices.update(recs_filtered)

    catalog_coverage = len(all_recommended_indices) / len(df)

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "catalog_coverage": float(catalog_coverage),
        "diversity": float(np.mean(diversities)) if diversities else 0.0
    }
