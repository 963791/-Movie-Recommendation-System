# 🎬 Movie Recommendation System

This project is a simple and effective Movie Recommendation System built using **Content-Based Filtering**. It uses **TF-IDF vectorization** and **cosine similarity** to recommend similar movies based on the overview description.

---

## 💡 How It Works

- Extracts textual information (overview) from the movie dataset
- Applies TF-IDF to convert text to numerical vectors
- Computes similarity using cosine distance
- Recommends the top 10 most similar movies

---

## 🧾 Dataset Used

The project uses files extracted from a public movie dataset:
- `movies_metadata.csv` — contains movie title, genres, overview, etc.
- `ratings_small.csv` — reserved for future collaborative filtering

---

## ⚙️ Features

- Recommend similar movies based on content
- Fast, interactive interface using Streamlit
- Lightweight and easy to deploy

---

## 🧪 Example Input

Movie Entered: **The Matrix**

Recommended Movies:
1. The Matrix Reloaded  
2. Inception  
3. Blade Runner  
4. Total Recall  
5. Interstellar  
... and more

---

## 🧰 How to Run the Project Locally

1. Clone the project folder
2. Make sure Python and pip are installed
3. Install dependencies using:
