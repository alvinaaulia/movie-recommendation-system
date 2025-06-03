
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

| Nama File | Deskripsi | Jumlah Baris |
|-----------|-----------|--------------|
| `movie.csv` | berisi metadata film: `movieId`, `title`, `genres` | 27.278 |
| `rating.csv` | rating dari pengguna: `userId`, `movieId`, `rating`, `timestamp` | 20.000.263 |
| `tag.csv` | tag yang diberikan user: `userId`, `movieId`, `tag`, `timestamp` | 465.564 |

Deskripsi fitur penting:
- `movieId`: ID unik film
- `title`: judul film
- `genres`: genre film dipisahkan dengan tanda `|`
- `userId`: ID unik pengguna
- `rating`: skor rating (0.5 - 5.0)
- `tag`: anotasi/komentar pendek pengguna
- `timestamp`: waktu ketika user memberikan rating/tag dalam format UNIX timestamp

## Exploratory Data Analysis (EDA)

Distribusi rating:
- Rating 4.0 dan 3.5 paling banyak diberikan
- Tidak ada missing value pada `rating.csv`
- Genre dan tag sangat bervariasi, relevan untuk content-based filtering

## Data Preparation
Langkah-langkah persiapan data yang dilakukan sebelum modeling:

1. **Gabungkan genre dan tag menjadi satu kolom `combined`**
   - Menggunakan `movie.csv` dan `tag.csv`, kita gabungkan kolom `genres` dan tag menjadi satu string per film.
2. **Bersihkan nilai NaN dari tag**
   - Film tanpa tag akan tetap dipertahankan, tetapi tag kosong dibuang dari `tag.csv` untuk menjaga kualitas konten.
3. **Encoding ID (Collaborative Filtering)**
   - `userId` dan `movieId` diubah menjadi indeks numerik dengan dictionary mapping.
4. **Normalisasi rating**
   - Semua rating dinormalisasi ke skala 0–1 agar sesuai dengan fungsi aktivasi sigmoid.
5. **TF-IDF Vectorization (Content-Based)**
   - Kolom `combined` ditransformasi menjadi vektor menggunakan `TfidfVectorizer`.
6. **Train-test split**
   - Data collaborative filtering dibagi menjadi 80% training dan 20% validasi.

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
- "Toy Story (1995)" → direkomendasikan "Toy Story 2", "Bug’s Life", dll.

    Contoh rekomendasi berdasarkan film "Toy Story (1995)":
    
    | Judul Film                 | Skor Kemiripan |
    |---------------------------|----------------|
    | Toy Story 2               | 0.936          |
    | A Bug's Life              | 0.812          |
    | Monsters, Inc.            | 0.792          |
    | ...                       | ...            |

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

    Contoh output Top-10 untuk `userId = 123`:
    
    | Judul Film                        | Predicted Rating |
    |----------------------------------|------------------|
    | The Shawshank Redemption         | 4.15             |
    | The Godfather                    | 4.12             |
    | Forrest Gump                     | 4.08             |
    | ...                              | ...              |

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
Average Precision@10: 0.0022
Average Recall@10: 0.0056
Average NDCG@10: 0.0107
```

- Average Precision@10 sebesar 0.0022 menunjukkan bahwa dari 10 film yang direkomendasikan, rata-rata hanya 2.2% yang pernah ditonton user sebelumnya. Ini berarti, dari total 900 film (90 user × 10 film), hanya sekitar 2 film yang cocok dengan histori user.
- Average Recall@10 sebesar 0.0056 mengindikasikan bahwa hanya 0.56% dari semua film yang pernah ditonton oleh user berhasil ditangkap oleh sistem dalam 10 rekomendasi teratas.
- Average NDCG@10 bernilai 0.0107, menunjukkan bahwa urutan rekomendasi sangat jauh dari ideal. Nilai ideal dari NDCG adalah 1.0 jika semua film yang disukai user direkomendasikan secara berurutan dari paling relevan.

**Kesimpulan:**
Nilai evaluasi yang rendah ini cukup wajar untuk sistem Content-Based Filtering. Model ini memang tidak dirancang untuk mereplikasi histori user secara langsung, melainkan untuk menemukan film baru yang mirip dengan yang disukai user sebelumnya. Fokusnya lebih pada kemiripan konten (genre/tag) ketimbang preferensi eksplisit pengguna.

