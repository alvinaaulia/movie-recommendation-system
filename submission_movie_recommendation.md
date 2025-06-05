
# Laporan Proyek Machine Learning - Movie Recommendation System

## Project Overview

Rekomendasi film menjadi salah satu fitur penting dalam layanan streaming modern seperti Netflix, Disney+, dan Amazon Prime. Dengan volume konten yang terus meningkat, pengguna sering merasa kewalahan memilih film yang sesuai preferensi mereka. Oleh karena itu, sistem rekomendasi yang akurat menjadi sangat krusial.

Proyek ini bertujuan membangun sistem rekomendasi film dengan dua pendekatan: **Content-Based Filtering** dan **Collaborative Filtering**, menggunakan **MovieLens 20M Dataset** dari GroupLens.

📚 Referensi:
- GroupLens. MovieLens 20M Dataset. [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
- Ricci, F., Rokach, L., Shapira, B. (2015). *Recommender Systems Handbook*. Springer.

## Business Understanding

#### Problem Statements

- Pengguna kesulitan menemukan film yang relevan di antara ribuan pilihan.
- Sistem rekomendasi konvensional kurang akurat atau tidak relevan karena hanya mengandalkan data dasar.

#### Goals

- Membuat sistem rekomendasi berbasis konten film.
- Mengembangkan model collaborative filtering berbasis interaksi pengguna.
- Menyediakan rekomendasi film personal dengan akurasi tinggi.

#### Solution Statements

- Menggunakan **Content-Based Filtering** dengan TF-IDF + Cosine Similarity dari genre dan tag.
- Menggunakan **Collaborative Filtering** berbasis Neural Network untuk mempelajari interaksi pengguna dan film melalui embedding.

## Data Understanding

Dataset yang digunakan adalah [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) yang terdiri dari beberapa file utama:

| Nama File | Deskripsi | Jumlah Baris | Jumlah Data Unik | Jumlah Kolom |
|-----------|-----------|--------------|------------------|--------------|
| `movie.csv` | berisi metadata film: `movieId`, `title`, `genres` | 27.278 | 27278 | 3 |
| `rating.csv` | rating dari pengguna: `userId`, `movieId`, `rating`, `timestamp` | 20.000.263 | 138493 | 4 |
| `tag.csv` | tag yang diberikan user: `userId`, `movieId`, `tag`, `timestamp` | 465.564 | 7801 | 4 |
| `link.csv` | identifier link di sumber lain: `movieId`, `imdbId`, `tmdbId` | 27.278 | 27278 | 3 |
| `genome_scores.csv` | nilai relevansi tags: `movieId`, `tagId`, `relevance` | 11.709.768 | 10381 | 3 |
| `genome_tags.csv` | deskripsi tags: `userId`, `movieId`, `tag`, `timestamp` | 1128 | 1128 | 4 |

Deskripsi fitur penting:
- `movieId`: ID unik film
- `title`: judul film
- `genres`: genre film dipisahkan dengan tanda `|`
- `userId`: ID unik pengguna
- `rating`: skor rating (0.5 - 5.0)
- `tag`: anotasi/komentar pendek pengguna
- `timestamp`: waktu ketika user memberikan rating/tag dalam format UNIX timestamp

Deskripsi fitur tambahan:
- `imdbId` : identifier (kode unik) sumber film di Internet Movie Database ID
- `tmdbId` : identifier (kode unik) sumber film di The Movie Database ID
- `relevance` : nilai relevansi tag terhadap film
- `tagId` : ID unik tag

## Exploratory Data Analysis (EDA)

Distribusi rating:
- Rating 4.0 dan 3.5 paling banyak diberikan
- Tidak ada missing value pada `rating.csv`
- Genre dan tag sangat bervariasi, relevan untuk content-based filtering

## Data Preparation
Langkah-langkah persiapan data yang dilakukan sebelum modeling:

1. **Mengatasi Missing Value**
   - Film tanpa tag akan tetap dipertahankan, tetapi tag kosong (16) dibuang dari `tag.csv` atau dataframe tags untuk menjaga kualitas konten.
2. **Mengatasi Duplikasi**
   - Tidak ada duplikasi pada `tags`.
3. **Agregasi data berdasarkan movieId**
   - Melakukan pengelompokan (grouping) dan penggabungan (aggregation) data berdasarkan movieId.
3. **Penggabungan Tag per Film**
   - Menggunakan `movie.csv` dan `tag.csv`, kita gabungkan kolom `genres` dan tag menjadi satu string per film (`movie_tags`)
   - Penggantian karakter pada genre pada kolom `movies_tags`.
   - Fungsi `fillna('')` digunakan untuk menangani nilai missing (NaN) pada kolom tag sebelum penggabungan menjadi kolom `combined`.
5. **TF-IDF Vectorization (Content-Based)**
   - Kolom `combined` ditransformasi menjadi vektor menggunakan `TfidfVectorizer`.
6. **Pengambilan sampel (sampling) dari data rating**
   - Pengambilan sebanyak 10% dari data asli (2000.000) digunakan agar tidak berat pada model tapi tetap efektif untuk training model Collaborative Filtering.
7. **Encoding ID (Collaborative Filtering)**
   - `userId` dan `movieId` diubah menjadi indeks numerik dengan dictionary mapping.
   - Semua rating dinormalisasi ke skala 0–1 agar sesuai dengan fungsi aktivasi sigmoid.
8. **Train-test split**
   - Data collaborative filtering dibagi menjadi 80% (160.000) training dan 20% (40.000) validasi.

## Modeling

#### A. Content-Based Filtering
- Cosine similarity digunakan untuk menghitung kesamaan antar film
- Fungsi:
```python
def movie_recommendations(title, k=5):
    sim_scores = cosine_sim_df[title].sort_values(ascending=False).iloc[1:k+1]
    return sim_scores
```
Contoh output: 

Rekomendasi berdasarkan film "Toy Story (1995)":

|          title         |        |
|------------------------|--------|
| Toy Story 2 (1999) |	0.925746 |
| Bug's Life, A (1998)	| 0.825143 |
| Ice Age (2002)	| 0.759788 |
| Monsters, Inc. (2001) | 0.747811 |
| Toy Story 3 (2010)	| 0.720269 |

#### B. Collaborative Filtering
- Menggunakan model embedding user dan movie
- Model dikembangkan dengan TensorFlow
- Arsitektur:
```python
class RecommenderNet(tf.keras.Model):
    ...
    def call(self, inputs):
        ...
```
- Metrik: Binary Crossentropy + RMSE
- Output: top-N rekomendasi untuk user tertentu berdasarkan prediksi rating tertinggi

Contoh output:  

Top 10 Rekomendasi untuk User 136903:
|   | title	| genres |	predicted_rating |
|---|-------|--------|-------------------|
| 316	| Shawshank Redemption, The (1994)	| Crime\|Drama |	4.148845 |
| 665	| World of Apu, The (Apur Sansar) (1959)	| Drama	| 4.102694 |
| 8696 | Cosmos (1980)	| Documentary	| 4.090828 |
| 2849 | Children of Paradise (Les enfants du paradis) ... | Drama\|Romance | 4.050789 |
| 2287 | Nights of Cabiria (Notti di Cabiria, Le) (1957)	| Drama	| 4.043172 |
| 3354 | Double Indemnity (1944)	| Crime\|Drama\|Film-Noir | 4.040033 |
| 6879 | Passion of Joan of Arc, The (Passion de Jeanne...	| Drama	| 4.037985 |
| 664	| Song of the Little Road (Pather Panchali) (1955)	| Drama | 4.027080 |
| 49	| Usual Suspects, The (1995) | Crime\|Mystery\|Thriller | 4.024944 |
| 851	| Godfather, The (1972)	| Crime\|Drama	| 4.020790 |

#### Kelebihan & Kekurangan Ke-2 Model:

| Metode               | Kelebihan                            | Kekurangan                            |
|----------------------|--------------------------------------|----------------------------------------|
| Content-Based        | Cepat, tidak perlu data user         | Tidak adaptif terhadap preferensi user |
| Collaborative        | Personalisasi tinggi                 | Butuh banyak data, cold-start problem  |

## Evaluation
### Collaborative Filtering
- **Metrik**: **RMSE (Root Mean Squared Error)**
  ![Image RMSE](https://github.com/alvinaaulia/dashboard-streamlit/blob/main/download%20(44).png)
- Training RMSE: 0.183
- Validation RMSE: 0.195
- Model tidak overfitting dan menunjukkan kemampuan generalisasi yang baik.

### Content-Based Filtering
Untuk mengevaluasi kinerja model Content-Based Filtering (CBF) secara lebih menyeluruh, digunakan metrik Precision@K, Recall@K, dan NDCG@K dengan nilai K=10. Evaluasi dilakukan pada 100 user acak dari dataset, dengan hasil sebagai berikut:
```
Evaluasi rata-rata untuk 90 user:
Average Precision@10: 0.0044
Average Recall@10: 0.0061
Average NDCG@10: 0.0171
```

- Average Precision@10 bernilai 0.0044 yang artinya dari 10 film yang direkomendasikan, hanya sekitar 0.044 (4.4%) yang pernah ditonton oleh user. 
- Average Recall@10 bernilai 0.0061 yang artinya dari semua film yang pernah ditonton oleh user, hanya sekitar 0.61% yang berhasil direkomendasikan oleh sistem (dalam 10 film teratas).
- Average NDCG@10 bernilsi 0.0171 yang artinya urutan rekomendasi kamu hampir tidak relevan dengan minat user. Nilai NDCG ideal = 1.0 jika semua film yang direkomendasikan adalah film yang disukai user, diurutkan dengan baik.

**Kesimpulan:**
Nilai evaluasi yang rendah ini cukup wajar untuk sistem Content-Based Filtering. Model ini memang tidak dirancang untuk mereplikasi histori user secara langsung, melainkan untuk menemukan film baru yang mirip dengan yang disukai user sebelumnya. Fokusnya lebih pada kemiripan konten (genre/tag) ketimbang preferensi eksplisit pengguna.

