import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download vader_lexicon if not already available
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

def add_sentiment(df):
    """
    Computes VADER sentiment for the entire dataframe.
    VADER is fast and suitable for dataset-wide calculations on CPU.
    """
    # Compute polarity using compound score
    df['sentiment_polarity'] = df['description'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    df['sentiment'] = df['sentiment_polarity'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )
    return df

# Global placeholder for lazy-loaded transformer pipeline
_transformer_pipeline = None

def get_transformer_pipeline():
    """
    Lazy loads Hugging Face's pipeline for DistilBERT sentiment analysis.
    This avoids loading weights at startup unless requested.
    """
    global _transformer_pipeline
    if _transformer_pipeline is None:
        from transformers import pipeline
        # Use default sentiment analysis pipeline, which uses DistilBERT base uncased finetuned sst-2
        _transformer_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    return _transformer_pipeline

def analyze_sentiment_transformer(text):
    """
    Analyzes sentiment of a single piece of text using DistilBERT.
    Returns: (label, score)
    """
    classifier = get_transformer_pipeline()
    result = classifier(text)[0]
    # Label is 'POSITIVE' or 'NEGATIVE'
    label = result['label'].title()  # 'Positive' or 'Negative'
    score = result['score']
    return label, score

def analyze_sentiment_vader(text):
    """
    Analyzes sentiment of a single piece of text using VADER.
    Returns: (label, score)
    """
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    label = 'Positive' if compound >= 0.05 else ('Negative' if compound <= -0.05 else 'Neutral')
    return label, compound
