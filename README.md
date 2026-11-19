# OpenCV Face Analysis

Dua program Python untuk mendeteksi wajah dengan Haar Cascade dan memperkirakan kelompok usia serta label wajah dengan model Caffe pralatih. Keduanya mendukung foto dan webcam.

## Instalasi

Gunakan Python 3.10–3.13. Dari terminal PowerShell di folder repo, jalankan:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe siapkan_aset.py
```

Skrip terakhir mengunduh foto contoh, konfigurasi, dan bobot model (sekitar 92 MB). Untuk macOS atau Linux, gunakan `./.venv/bin/python` sebagai pengganti `.\.venv\Scripts\python.exe`.

## Uji dengan foto

```powershell
.\.venv\Scripts\python.exe face_detection.py --image sample.jpg --output hasil-wajah.png
.\.venv\Scripts\python.exe age_gender_estimation.py --image sample.jpg --output hasil-klasifikasi.png
```

Di VS Code, kedua perintah tersedia melalui **Tasks: Run Task** pada Windows.

## Jalankan dengan webcam

```powershell
.\.venv\Scripts\python.exe face_detection.py
.\.venv\Scripts\python.exe age_gender_estimation.py --nama "Nama Kamu"
```

Tekan **Q** atau **Esc** untuk keluar. Gunakan `--camera 1` untuk memilih kamera indeks 1. Estimasi usia dan label merupakan keluaran model berdasarkan pola visual.

## Sumber aset

- [Bobot model Caffe oleh Levi dan Hassner](https://github.com/GilLevi/AgeGenderDeepLearning)
- [Konfigurasi model dari LearnOpenCV](https://github.com/spmallick/learnopencv/tree/master/AgeGender)
- [Foto contoh dari OpenCV](https://github.com/opencv/opencv/blob/master/samples/data/lena.jpg)
