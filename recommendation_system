# 1 - Import Library
import pandas as pd
from scipy import sparse

# 2 - Import Data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# 3 - Baca dataset ratings
print(ratings.head())

# 4 - Cek dan hapus data null di ratings
print(ratings.isnull().sum())  # cek jumlah null
ratings.dropna(inplace=True)  # hapus baris dengan nilai kosong

# 5 - Baca dataset movies
print(movies.head())

# 6 - Cek dan hapus data null di movies
print(movies.isnull().sum())  # cek jumlah null
movies.dropna(inplace=True)   # hapus baris kosong

# 7 - Cek ukuran dataset ratings
ratings.shape

# 8 - Cek ukuran dataset movies
movies.shape

# 9 - Atribut yang sama antar kedua dataset
# Kolom movieId adalah kunci relasi antara ratings.csv dan movies.csv

# 10 - Merge dataset ratings dan movies
ratings = pd.merge(movies, ratings).drop(['genres', 'timestamp'], axis=1)
print(ratings.shape)

# Pivot data ke bentuk matriks user-rating
userRatings = ratings.pivot_table(index='userId', columns='title', values='rating')
print(userRatings)

# 11 - Bersihkan matriks: hapus film yang dirating <10 user
print("Before: ", userRatings.shape)
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0)
print("After: ", userRatings.shape)

# 12 - Hitung matriks korelasi antar film
corrMatrix = userRatings.corr(method='pearson')
corrMatrix

# 13 - Buat daftar film favorit bertema Romance
romantic_lover = [
    ("(500) Days of Summer (2009)", 5),
    ("Alice in Wonderland (2010)", 3),
    ("Aliens (1986)", 1),
    ("2001: A Space Odyssey (1968)", 2)
]

# 14 - Fungsi untuk mencari film serupa
def get_similar(movie_name, rating):
    similar_ratings = corrMatrix[movie_name] * rating
    similar_ratings = similar_ratings.sort_values(ascending=False)
    return similar_ratings

similar_movies = pd.concat([get_similar(movie, rating) for movie, rating in romantic_lover], axis=1)
similar_movies.head(10)

# 15 - Buat genre lain: Comedy dan Fungsi untuk mencari film serupa
comedy_lover = [
    ("Toy Story (1995)", 4),
    ("Grumpier Old Men (1995)", 5),
    ("Get Shorty (1995)", 4),
    ("Father of the Bride Part II (1995)", 5)
]

similar_comedy = pd.concat([get_similar(movie, rating) for movie, rating in comedy_lover], axis=1)
similar_comedy.head(10)
