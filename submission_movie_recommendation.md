
# Laporan Proyek Machine Learning - Movie Recommendation System

## Project Overview

Rekomendasi film menjadi salah satu fitur penting dalam layanan streaming modern seperti Netflix, Disney+, dan Amazon Prime. Dengan volume konten yang terus meningkat, pengguna sering merasa kewalahan memilih film yang sesuai preferensi mereka. Oleh karena itu, sistem rekomendasi yang akurat menjadi sangat krusial.

Proyek ini bertujuan membangun sistem rekomendasi film dengan dua pendekatan: **Content-Based Filtering** dan **Collaborative Filtering**, menggunakan **MovieLens 20M Dataset** dari GroupLens.

📚 Referensi:
- GroupLens. MovieLens 20M Dataset. [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
- Ricci, F., Rokach, L., Shapira, B. (2015). *Recommender Systems Handbook*. Springer.

## Business Understanding

### Problem Statements

- Pengguna kesulitan menemukan film yang relevan di antara ribuan pilihan.
- Sistem rekomendasi konvensional kurang akurat atau tidak relevan karena hanya mengandalkan data dasar.

### Goals

- Membuat sistem rekomendasi berbasis konten film.
- Mengembangkan model collaborative filtering berbasis interaksi pengguna.
- Menyediakan rekomendasi film personal dengan akurasi tinggi.

### Solution Statements

- Menggunakan **Content-Based Filtering** dengan TF-IDF + Cosine Similarity dari genre dan tag.
- Menggunakan **Collaborative Filtering** berbasis Neural Network untuk mempelajari interaksi pengguna dan film melalui embedding.

## Data Understanding

Dataset yang digunakan adalah [MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset). Terdapat lebih dari 20 juta rating dan ribuan metadata film. File utama:
- `movie.csv`: metadata film
- `rating.csv`: interaksi rating user-film
- `tag.csv`: tag yang diberikan user
- `link.csv`, `genome_scores.csv`, `genome_tags.csv`: fitur tambahan, tidak digunakan.

Variabel penting:
- `movieId`, `title`, `genres`, `userId`, `rating`, `tag`

Jumlah data:
- 19.545 film
- 138.000+ user
- 20 juta interaksi rating
- 1 juta+ tag

### Exploratory Data Analysis (EDA)

Distribusi rating:
- Rating 4.0 dan 3.5 paling banyak diberikan
- Tidak ada missing value pada `rating.csv`
- Genre dan tag sangat bervariasi, relevan untuk content-based filtering

## Data Preparation

Langkah-langkah yang dilakukan:
1. Gabungkan `genres` dan `tag` → `combined`
2. Hapus nilai NaN dari `tag`
3. Sampling 10% dari `rating.csv` untuk efisiensi
4. Normalisasi nilai rating untuk collaborative filtering (0–1)
5. Split data menjadi training dan validation (80:20)

Tujuan preparation:
- Menghapus noise (missing/duplicated)
- Membuat fitur siap pakai untuk pemodelan (TF-IDF dan vektor input)

## Modeling

### A. Content-Based Filtering
- Menggunakan TF-IDF dari fitur `combined`
- Cosine similarity digunakan untuk menghitung kesamaan antar film
- Fungsi:
```python
def movie_recommendations(title, k=5):
    sim_scores = cosine_sim_df[title].sort_values(ascending=False).iloc[1:k+1]
    return sim_scores
```
Contoh output:
- "Toy Story (1995)" → direkomendasikan "Toy Story 2", "Bug’s Life", dll.

### B. Collaborative Filtering
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

Kelebihan & Kekurangan:

| Metode               | Kelebihan                            | Kekurangan                            |
|----------------------|--------------------------------------|----------------------------------------|
| Content-Based        | Cepat, tidak perlu data user         | Tidak adaptif terhadap preferensi user |
| Collaborative        | Personalisasi tinggi                 | Butuh banyak data, cold-start problem  |

## Evaluation

### Metrik Evaluasi

- **RMSE (Root Mean Squared Error)**
![Image RMSE](https://github.com/alvinaaulia/dashboard-streamlit/blob/main/download%20(44).png)

### Hasil

- RMSE Training: ~0.183
- RMSE Validation: ~0.195
- Loss stabil sejak epoch ke-5 → tidak overfitting

Evaluasi memperlihatkan bahwa model collaborative filtering dapat memprediksi rating dengan baik dan content-based menghasilkan rekomendasi logis sesuai genre/tag.
