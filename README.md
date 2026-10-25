# BelimbingCheck — Website Klasifikasi Kesegaran Belimbing

Prototipe web untuk mengintegrasikan model `best_model.keras` ke aplikasi Flask dengan UI responsif berbahasa Indonesia.

## Fitur
- Unggah foto JPG/JPEG/PNG/WebP (maks. 8 MB)
- Pratinjau gambar sebelum dikirim
- Prediksi 3 kelas: **BELIMBING BUSUK**, **BELIMBING SEGAR**, **BELIMBING TIDAK SEGAR**
- Menampilkan kelas prediksi, confidence, dan distribusi probabilitas
- Tampilan responsif desktop dan ponsel

## Menjalankan di Windows
Disarankan menggunakan Python 3.10 atau 3.11 dan virtual environment.

1. Ekstrak ZIP ini.
2. Buka terminal di folder `belimbing_freshness_web`.
3. Buat environment dan aktifkan:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

4. Buka `http://127.0.0.1:5000` di browser.

Jika TensorFlow tidak cocok dengan versi Python/Windows yang digunakan, coba environment Python 3.10/3.11 atau jalankan melalui WSL/Linux.

## Penting: kesesuaian label
Notebook memakai `flow_from_directory`, yang mengurutkan nama folder kelas secara alfabetis. Aplikasi ini menggunakan urutan:
1. `BELIMBING BUSUK`
2. `BELIMBING SEGAR`
3. `BELIMBING TIDAK SEGAR`

Urutan tersebut **harus** sesuai dengan `train_generator.class_indices` saat model dilatih. Jika urutan indeks pada model berbeda, ubah `CLASS_NAMES` pada `app.py` sesuai hasil notebook sebelum digunakan.

## Preprocessing
Aplikasi mengikuti notebook: gambar diubah menjadi RGB, di-resize ke 224×224, lalu diproses dengan `tensorflow.keras.applications.mobilenet_v3.preprocess_input`.

## Catatan penelitian dan HKI
- UI ini merupakan implementasi awal/prototipe, bukan jaminan bahwa hasil prediksi valid untuk semua kondisi foto.
- Sebelum demonstrasi/pendaftaran HKI, lakukan uji integrasi dengan beberapa foto terpisah, verifikasi urutan label model, dokumentasikan hasil pengujian, dan cantumkan atribusi/asal dataset serta lisensi aset yang digunakan.
- Jangan mengunggah aplikasi ke internet publik tanpa menambahkan pembatasan akses, HTTPS, konfigurasi secret key, logging yang aman, dan kebijakan retensi/penghapusan foto.
- Font Google digunakan sebagai peningkatan tampilan; bila ingin aplikasi sepenuhnya offline, hapus `@import`/tautan Google Fonts dan gunakan font sistem.
