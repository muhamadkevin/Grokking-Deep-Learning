# Grokking Deep Learning — Bab 1–6

| | |
|---|---|
| **Nama** | Muhamad Kevin |
| **NIM** | 101032300243 |
| **Kelas** | BS1TK-47-REG-G13 |

Repositori ini berisi rangkuman dan implementasi kode Python untuk **Bab 1 sampai Bab 6** dari buku *Grokking Deep Learning* (Andrew W. Trask, Manning, 2019). Setiap bab disajikan dalam satu Jupyter Notebook yang menggabungkan penjelasan konsep (Bahasa Indonesia) dengan contoh kode, visualisasi, latihan, dan contoh solusinya.

Sesuai filosofi buku, semua neural network dibangun **dari nol hanya dengan NumPy** — mulai dari satu bobot, gradient descent, hingga deep neural network pertama dengan backpropagation.

---

## Daftar Isi

1. [Struktur Proyek](#struktur-proyek)
2. [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
3. [Bab 1 — Introducing Deep Learning](#bab-1--introducing-deep-learning)
4. [Bab 2 — Fundamental Concepts](#bab-2--fundamental-concepts)
5. [Bab 3 — Forward Propagation](#bab-3--forward-propagation)
6. [Bab 4 — Gradient Descent](#bab-4--gradient-descent)
7. [Bab 5 — Generalizing Gradient Descent](#bab-5--generalizing-gradient-descent)
8. [Bab 6 — Backpropagation](#bab-6--backpropagation)
9. [Dataset](#dataset)
10. [Referensi](#referensi)

---

## Struktur Proyek

```
Grokking Deep Learning/
├── README.md
└── notebooks/
    ├── 01_Introducing_Deep_Learning.ipynb
    ├── 02_Fundamental_Concepts.ipynb
    ├── 03_Forward_Propagation.ipynb
    ├── 04_Gradient_Descent.ipynb
    ├── 05_Generalizing_Gradient_Descent.ipynb
    ├── 06_Backpropagation.ipynb
    └── data/
        └── mnist.npz         # dataset MNIST (dipakai di Bab 5)
```

## Instalasi & Cara Menjalankan

```bash
pip install numpy matplotlib jupyter
cd notebooks
jupyter notebook
```

Bab 1–4 dan 6 memakai data mainan yang didefinisikan langsung di notebook. Bab 5 memakai MNIST melalui fungsi `fetch()`, yang membaca `notebooks/data/mnist.npz` dan otomatis mengunduhnya dari mirror resmi Keras jika file belum ada.

---

## Bab 1 — Introducing Deep Learning

📓 [`01_Introducing_Deep_Learning.ipynb`](notebooks/01_Introducing_Deep_Learning.ipynb)

Bab motivasi: **mengapa** deep learning layak dipelajari, **mengapa** buku ini berbeda, dan **apa** yang dibutuhkan untuk mulai.

**Materi:**
1. Selamat datang & arti kata *grok*
2. Tiga alasan belajar deep learning & contoh penerapannya
3. Apakah sulit dipelajari? (*fun payoff*)
4. Mengapa membaca buku ini (analogi NASCAR & palu)
5. Yang dibutuhkan: Jupyter, NumPy, matematika SMA, masalah pribadi, Python
6. Pemanasan Python (list, loop, fungsi, `w_sum`)
7. Pemanasan NumPy & benchmark vektorisasi
8. Matematika SMA yang dibutuhkan (perkalian, grafik x–y, slope)
9. Sekilas neural network yang **belajar**
10. Peta perjalanan buku

**Rangkuman:**
- Deep learning adalah alat yang kuat untuk **otomatisasi kecerdasan secara bertahap** dan menyenangkan untuk dipelajari.
- Pendekatan buku: **intuisi + kode dari nol**, matematika setingkat SMA.
- Vektorisasi NumPy memberi hasil sama dengan loop Python tetapi jauh lebih cepat.

---

## Bab 2 — Fundamental Concepts

📓 [`02_Fundamental_Concepts.ipynb`](notebooks/02_Fundamental_Concepts.ipynb)

*How do machines learn?* — memetakan lanskap machine learning, disertai implementasi kecil setiap konsep.

**Materi:**
1. Deep learning ⊂ machine learning ⊂ AI
2. Machine learning: *monkey see, monkey do* (diprogram eksplisit vs belajar dari contoh)
3. Supervised learning (Monday → Tuesday stock prices)
4. Unsupervised learning (clustering *puppies, pizza, kittens, hot dog, burger*; k-means)
5. Parametric vs nonparametric (analogi pasak persegi)
6. Supervised parametric learning: *predict → compare → learn* (mesin kenop Red Sox)
7. Unsupervised parametric learning (peluang keanggotaan kelompok)
8. Nonparametric learning (counting lampu lalu lintas, k-NN)
9. Empat kategori algoritma

**Rangkuman:**
- **Supervised**: *apa yang diketahui* → *apa yang ingin diketahui*; **unsupervised**: mengelompokkan data tanpa label.
- **Parametric**: jumlah parameter tetap (*trial and error* memutar kenop); **nonparametric**: parameter ditentukan data (*counting*).
- Deep learning adalah **parametric learning** dengan siklus **predict → compare → learn**.

---

## Bab 3 — Forward Propagation

📓 [`03_Forward_Propagation.ipynb`](notebooks/03_Forward_Propagation.ipynb)

Langkah pertama siklus belajar: **predict**.

**Materi:**
1. Neural network 1 input → 1 output & tiga cara memandang bobot
2. Multiple inputs — weighted sum & kontribusi tiap input
3. Dot product sebagai ukuran kemiripan (AND/NOT lunak)
4. Multiple outputs — elementwise multiplication
5. Multiple inputs & outputs — vector-matrix multiplication
6. Predicting on predictions — hidden layer (Python murni & NumPy)
7. Primer NumPy (broadcasting, reshape, axis, aturan shape)

**Rangkuman:**
- **Forward propagation** = mengalirkan input melalui bobot untuk menghasilkan prediksi.
- **Dot product** mengukur **kemiripan** input dengan bobot; kontribusi = input × bobot.
- Layer bisa **ditumpuk**; perhatikan aturan shape `(m,n)·(n,k) → (m,k)`.

---

## Bab 4 — Gradient Descent

📓 [`04_Gradient_Descent.ipynb`](notebooks/04_Gradient_Descent.ipynb)

Langkah **compare** dan **learn**: algoritma belajar terpenting dalam deep learning.

**Materi:**
1. Mean squared error & mengapa error dikuadratkan
2. Hot and cold learning & kelemahannya (step size tetap)
3. Arah & besar perubahan: efek *stopping*, *negative reversal*, *scaling*
4. Satu iterasi & beberapa langkah gradient descent
5. *Tunnel vision* & *a box with rods*: derivative secara intuitif
6. Memakai derivative untuk belajar (+ verifikasi numerik)
7. Overcorrection & divergence
8. Alpha (learning rate) & pencarian alpha berdasarkan orde besaran

**Rangkuman:**
- `delta = pred - goal`, `weight_delta = delta * input`, `weight -= alpha * weight_delta`.
- `weight_delta` adalah **derivative** error terhadap bobot — slope kurva error.
- Input besar bisa menyebabkan **divergence**; diatasi dengan **alpha**.

---

## Bab 5 — Generalizing Gradient Descent

📓 [`05_Generalizing_Gradient_Descent.ipynb`](notebooks/05_Generalizing_Gradient_Descent.ipynb)

Gradient descent untuk **banyak bobot sekaligus**.

**Materi:**
1. Multiple inputs (satu delta, banyak weight delta)
2. Mengamati langkah learning & pentingnya normalisasi input
3. Freezing one weight & pergeseran kurva error
4. Multiple outputs
5. Multiple inputs & outputs (outer product)
6. MNIST: training network 784 → 10, akurasi, confusion matrix
7. Visualisasi bobot sebagai template & visualisasi dot product

**Rangkuman:**
- Setiap bobot di-update dengan `delta_output × input_bobot`; `weight_deltas = outer(delta, input)`.
- Error ditentukan **bersama** oleh semua bobot; network berhenti belajar saat error = 0.
- Network satu layer di MNIST mencapai ±75% akurasi test; bobotnya adalah "template" tiap angka.

---

## Bab 6 — Backpropagation

📓 [`06_Backpropagation.ipynb`](notebooks/06_Backpropagation.ipynb)

Membangun **deep neural network pertama** pada *streetlight problem*.

**Materi:**
1. Streetlight problem & *matrix relationship* (encoding berbeda, pola sama)
2. Mempelajari seluruh dataset; full, batch, dan stochastic GD
3. Up and down pressure — network belajar korelasi
4. Edge case: overfitting & conflicting pressure
5. Indirect correlation & menciptakan korelasi dengan hidden layer
6. Backpropagation: *long-distance error attribution*
7. Linear vs nonlinear (ReLU) — *sometimes correlation*
8. Backpropagation dalam kode & verifikasi dengan gradien numerik
9. Versi full batch, pengaruh jumlah hidden node, mengapa deep network penting

**Rangkuman:**
- Jika tidak ada korelasi langsung, **hidden layer** dibutuhkan untuk **menciptakan korelasi**.
- Dua layer linear = satu layer linear → butuh **nonlinearity** (ReLU).
- **Backpropagation**: `layer_1_delta = layer_2_delta · W_12ᵀ ⊙ relu'(layer_1)`.

---

## Dataset

| Bab | Dataset | Sumber |
|---|---|---|
| 1–4, 6 | Data mainan (toes/wlrec/nfans, streetlight, dll.) | Didefinisikan di notebook |
| 5 | MNIST (`mnist.npz`) | Mirror resmi Keras |

## Library yang Digunakan

`numpy` · `matplotlib`

## Referensi

- Trask, A. W. (2019). *Grokking Deep Learning*. Manning Publications.
- Repositori kode & data resmi: <https://github.com/iamtrask/Grokking-Deep-Learning>
