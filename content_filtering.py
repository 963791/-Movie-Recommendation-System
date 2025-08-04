import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
metadata_path = 'Dataset/movies_metadata.csv'
df = pd.read_csv(metadata_path, low_memory=False).head(5000)

# Fill NaN overviews with empty string
df['overview'] = df['overview'].fillna('')

# TF-IDF Vectorizer on the overview
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reset index with movie titles for easier lookups
df = df.reset_index()
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        return f"Movie '{title}' not found in dataset."
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices]

# Example usage
if __name__ == "__main__":
    movie_title = "The Dark Knight"  # Change to any title from dataset
    recommendations = get_recommendations(movie_title)
    print(f"\nTop 10 recommendations for '{movie_title}':\n")
    print(recommendations)