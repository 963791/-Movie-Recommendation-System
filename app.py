import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/movies_metadata.csv", low_memory=False).head(5000)
    df['overview'] = df['overview'].fillna('')
    return df

df = load_data()

# Vectorize movie overviews
@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, pd.Series(df.index, index=df['title']).drop_duplicates()

cosine_sim, indices = compute_similarity(df)

# Recommendation function
def get_recommendations(title):
    idx = indices.get(title)
    if idx is None:
        return None
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.subheader("Get similar movies based on content!")

movie_input = st.text_input("Enter a movie title (e.g., Avatar)", "")

if movie_input:
    recommendations = get_recommendations(movie_input)
    if recommendations is not None:
        st.write("### 🔍 Top 10 Recommendations:")
        for i, title in enumerate(recommendations, 1):
            st.write(f"{i}. {title}")
    else:
        st.warning("Movie not found. Please try another title.")
