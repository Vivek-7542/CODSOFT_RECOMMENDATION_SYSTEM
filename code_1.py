import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
movies = pd.DataFrame({
    'title': [
        'Inception',
        'Interstellar',
        'The Dark Knight',
        'Avatar',
        'Titanic',
        'The Matrix',
        'John Wick',
        'Avengers Endgame'
    ],
    'genre': [
        'Sci-Fi Action',
        'Sci-Fi Drama',
        'Action Crime',
        'Sci-Fi Adventure',
        'Romance Drama',
        'Sci-Fi Action',
        'Action Thriller',
        'Action Sci-Fi'
    ]
})

# Convert genres into numerical vectors
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genre'])

# Calculate similarity
similarity = cosine_similarity(genre_matrix)

def recommend(movie_name):
    movie_name = movie_name.lower()

    if movie_name not in movies['title'].str.lower().values:
        print("Movie not found!")
        return

    index = movies[movies['title'].str.lower() == movie_name].index[0]

    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    print(f"\nMovies similar to '{movies.iloc[index]['title']}':\n")

    for i in distances[1:6]:
        print(movies.iloc[i[0]]['title'])

# Main Program
print("Available Movies:")
for movie in movies['title']:
    print("-", movie)

movie = input("\nEnter a movie name: ")
recommend(movie)