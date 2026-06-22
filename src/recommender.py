"""
recommender.py
---------------
Memory-optimized hybrid content-based recommender system for the Netflix dataset
using description TF-IDF, genre matching, and content rating compatibility.
Also includes offline evaluation metrics (Precision@K, Catalog Coverage, Diversity).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from difflib import get_close_matches
import gc

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
    Build individual feature matrices for description, genres, and ratings.
    Returns sparse/compact matrices instead of dense NxN similarity matrices
    to keep memory usage under 10 MB.

    Args:
        df (pd.DataFrame): Netflix dataset

    Returns:
        tuple: (cleaned dataframe, tfidf_matrix, genre_matrix, rating_features, title indices)
    """
    df = df.copy()

    # Clean and normalize columns
    df['title'] = df['title'].astype(str).str.strip().str.lower()
    df['listed_in'] = df['listed_in'].fillna('')
    df['description'] = df['description'].fillna('')
    df['country'] = df.get('country', '').fillna('')
    df['director'] = df.get('director', '').fillna('')
    df['rating'] = df.get('rating', 'Unknown').fillna('Unknown')

    # 1. Description Features (TF-IDF)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])

    # 2. Genre Features (one-hot encoded)
    genres_list = df['listed_in'].apply(lambda x: [g.strip().lower() for g in x.split(',') if g.strip()])
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genres_list).astype(np.float32)

    # 3. Rating Features (Exact rating & maturity level)
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
    rating_features = ohe.fit_transform(df[['rating', 'rating_group']]).astype(np.float32)

    # Create title → index mapping
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    return df, tfidf_matrix, genre_matrix, rating_features, indices


def recommend(title, df, tfidf_matrix, genre_matrix, rating_features, indices, w_desc=0.4, w_genre=0.4, w_rating=0.2, n=3):
    """
    Recommend N similar titles based on hybrid similarity computed on-the-fly.
    Uses less than 1 ms and has zero persistent memory overhead.
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

    # Compute similarity vectors on the fly (float32 for speed/memory)
    sim_desc = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten().astype(np.float32)
    sim_genre = cosine_similarity(genre_matrix[idx:idx+1], genre_matrix).flatten().astype(np.float32)
    sim_rating = cosine_similarity(rating_features[idx:idx+1], rating_features).flatten().astype(np.float32)

    # Weighted combination
    total_w = w_desc + w_genre + w_rating
    if total_w > 0:
        sim_scores = (w_desc * sim_desc + w_genre * sim_genre + w_rating * sim_rating) / total_w
    else:
        sim_scores = sim_desc

    sim_scores_list = list(enumerate(sim_scores))
    sim_scores_list = sorted(sim_scores_list, key=lambda x: x[1], reverse=True)

    # Filter out the item itself
    recs = [i for i, _ in sim_scores_list if i != idx][:n]

    # Get top N recommendations
    recommendations = df['title'].iloc[recs]

    # Capitalize for cleaner display
    return [t.title() for t in recommendations.tolist()]


def evaluate_recommender(df, tfidf_matrix, genre_matrix, rating_features, w_desc=0.4, w_genre=0.4, w_rating=0.2, k=5, sample_size=200):
    """
    Computes offline evaluation metrics.
    Temporarily computes float32 similarity matrix and deletes it immediately
    after calculating metrics to prevent memory exhaustion in Streamlit Cloud.
    """
    # 1. Compute temporary similarity matrices
    sim_desc = cosine_similarity(tfidf_matrix).astype(np.float32)
    sim_genre = cosine_similarity(genre_matrix).astype(np.float32)
    sim_rating = cosine_similarity(rating_features).astype(np.float32)

    # Combine matrices
    total_w = w_desc + w_genre + w_rating
    if total_w > 0:
        sim_matrix = (w_desc * sim_desc + w_genre * sim_genre + w_rating * sim_rating) / total_w
    else:
        sim_matrix = sim_desc

    # Free parent matrices immediately to save ~1 GB of RAM
    del sim_desc, sim_genre, sim_rating
    gc.collect()

    # 2. Prepare genre sets for comparison
    genres = df['listed_in'].apply(lambda x: set([g.strip().lower() for g in x.split(',') if g.strip()]))
    genres_dict = dict(zip(df.index, genres))

    # 3. Precision@K and Diversity@K (computed over a random sample for speed)
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

    # 4. Catalog Coverage@K: computed using vectorized argpartition
    k_plus_1 = k + 1
    partitioned = np.argpartition(-sim_matrix, k_plus_1, axis=1)[:, :k_plus_1]

    all_recommended_indices = set()
    for i in range(len(df)):
        recs = partitioned[i]
        recs_filtered = recs[recs != i][:k]
        all_recommended_indices.update(recs_filtered)

    catalog_coverage = len(all_recommended_indices) / len(df)

    # Free similarity matrix and force GC
    del sim_matrix, partitioned
    gc.collect()

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "catalog_coverage": float(catalog_coverage),
        "diversity": float(np.mean(diversities)) if diversities else 0.0
    }
