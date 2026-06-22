# 🎬 Netflix Dashboard

An interactive Streamlit dashboard that analyzes Netflix content data using **Data-driven insights**, including **sentiment analysis** and a **content-based recommendation engine**.

---

## 🚀 Features

### 📊 1. Dataset Overview
- Explore Netflix movies and TV shows.
- Filter by **Type**, **Country**, and **Genre**.
- Visualize content distribution using bar charts.

### 💬 2. Sentiment Analysis
- Dataset-wide sentiment classification of show descriptions using **VADER** (fast, lexicon-based).
- On-demand, high-fidelity sentiment prediction of custom user text with a toggle between **VADER** and **DistilBERT** (Hugging Face Transformer model).
- View distribution of positive, negative, and neutral sentiments across the catalog.

### 🎯 3. Hybrid Recommendation Engine
- Recommender combining **description semantics (TF-IDF)**, **genre overlaps (multi-label binarizer)**, and **content age rating compatibility**.
- Interactive weights selection via Streamlit sliders to adjust feature balance.
- **Offline Evaluation Metrics** showing **Precision @ 5**, **Catalog Coverage @ 5**, and **Recommendation Diversity @ 5** to measure model performance and diversity trade-offs.

### 📈 4. Trend Analysis
- Visualize Netflix library growth since 2008 (Movies vs. TV Shows added).
- Track the popularity evolution of the top 5 genres over the years.
- View geographic production distribution trends for top countries.

---

## 🖼️ Demo Screenshots

### Overview Tab
![Overview Tab](screenshots/overview.png)

### Sentiment Analysis Tab
![Sentiment Tab](screenshots/sentiment.png)

### Recommendation Tab
![Recommendation Tab](screenshots/recommendation.png)

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | [Streamlit](https://streamlit.io) |
| **Data Processing** | Pandas, Scikit-learn, Numpy |
| **Sentiment Analysis** | VADER (NLTK Lexicon), DistilBERT (Hugging Face Transformers / PyTorch) |
| **Machine Learning** | Hybrid Multi-Feature Cosine Similarity + Offline Evaluation Metrics |
| **Language** | Python 3.8+ |

---

## 🧩 Project Structure

```
netflix-ai-dashboard/
├── data/
│   └── netflix_titles.csv
├── screenshots/
│   ├── overview.png
│   ├── sentiment.png
│   └── recommendation.png
├── src/
│   ├── data_cleaning.py
│   ├── sentiment_analysis.py
│   └── recommender.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/netflix-dashboard.git
cd netflix-dashboard
```

### 2️⃣ Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Start the Netflix Dashboard locally:

```bash
streamlit run app.py
```

