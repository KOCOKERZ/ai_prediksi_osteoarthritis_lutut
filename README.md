# Penjelasan file app.py :

### Import Libraries

```python
import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt  
import time
import warnings
```

- `streamlit as st`: Mengimpor pustaka Streamlit untuk membuat antarmuka web.
- `tensorflow as tf`: Mengimpor TensorFlow untuk memuat dan menggunakan model pembelajaran mesin.
- `from PIL import Image, ImageOps`: Mengimpor modul untuk manipulasi gambar.
- `import numpy as np`: Mengimpor NumPy untuk operasi numerik.
- `import matplotlib.pyplot as plt`: Mengimpor Matplotlib untuk pembuatan plot (walaupun tidak digunakan dalam kode ini).
- `import time`: Mengimpor modul time untuk penundaan waktu.
- `import warnings`: Mengimpor modul warnings untuk mengabaikan peringatan yang tidak diinginkan.

### Mengabaikan Peringatan

```python
warnings.filterwarnings("ignore")
```

Mengabaikan semua peringatan yang mungkin muncul selama eksekusi kode.

### Konfigurasi Halaman Streamlit

```python
st.set_page_config(
    page_title="Prediksi Osteoarthritis pada Lutut",
    page_icon=":bone:",
    layout="centered",
    initial_sidebar_state='auto'
)
```

Mengatur konfigurasi halaman Streamlit:
- `page_title`: Judul halaman.
- `page_icon`: Ikon yang muncul di tab browser.
- `layout`: Tata letak halaman (tengah).
- `initial_sidebar_state`: Status awal sidebar (otomatis).

### Menyembunyikan Elemen Streamlit

```python
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1fcdlh1 {padding: 2rem 1rem 10rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
```

Menggunakan CSS untuk menyembunyikan menu utama dan footer Streamlit.

### Fungsi untuk Memproses dan Memprediksi Gambar

```python
def import_and_predict(image_data, model):
    img_size = 128
    img = ImageOps.grayscale(image_data)
    img = img.resize((img_size, img_size))
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction
```

- `img_size = 128`: Ukuran gambar yang diinginkan (128x128 piksel).
- `ImageOps.grayscale(image_data)`: Mengubah gambar menjadi grayscale.
- `img.resize((img_size, img_size))`: Mengubah ukuran gambar menjadi 128x128 piksel.
- `np.asarray(img) / 255.0`: Mengubah gambar menjadi array NumPy dan menormalisasi nilai piksel menjadi 0-1.
- `np.expand_dims(img, axis=-1)`: Menambahkan dimensi untuk saluran warna.
- `np.expand_dims(img, axis=0)`: Menambahkan dimensi untuk batch.
- `model.predict(img)`: Memprediksi kondisi gambar menggunakan model.

### Fungsi untuk Memuat Model

```python
@st.cache_resource
def load_model():
    model_path = 'model.h5'
    return tf.keras.models.load_model(model_path)
```

- `@st.cache_resource`: Dekorator untuk menyimpan hasil pemuatan model agar tidak perlu dimuat berulang kali.
- `model_path = 'model.h5'`: Jalur file model yang akan dimuat.
- `tf.keras.models.load_model(model_path)`: Memuat model TensorFlow dari file.

### Memuat Model dan Menentukan Kategori

```python
model = load_model()

categories = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']
descriptions = {
    'Normal': 'Lutut tampak normal tanpa tanda-tanda osteoartritis.',
    'Doubtful': 'Ada tanda-tanda minor yang mungkin menunjukkan tahap awal osteoartritis.',
    'Mild': 'Ada tanda-tanda ringan osteoartritis, dengan beberapa penyempitan ruang sendi.',
    'Moderate': 'Ada tanda-tanda sedang osteoartritis, dengan penyempitan ruang sendi yang terlihat dan kemungkinan adanya taji tulang.',
    'Severe': 'Ada tanda-tanda parah osteoartritis, dengan penyempitan ruang sendi yang signifikan dan taji tulang yang besar.'
}
```

- `model = load_model()`: Memuat model ke dalam variabel `model`.
- `categories`: Daftar kategori prediksi.
- `descriptions`: Kamus yang memberikan deskripsi untuk setiap kategori.

### Antarmuka Pengguna Streamlit

```python
st.title("Prediksi Osteoarthritis pada Lutut")

file = st.file_uploader("Unggah gambar X-ray lutut, dan AI akan memprediksi kondisinya serta memberikan deskripsi singkat.", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("Silakan unggah file gambar")
else:
    image = Image.open(file)
    st.image(image, caption='Gambar Terunggah', use_column_width=True)
    st.write("")
    st.write("Memprediksi...")
    
    with st.spinner('Tunggu sebentar...'):
        time.sleep(4)  
        predictions = import_and_predict(image, model)
        predicted_class = np.argmax(predictions)

    result_text = f"A.I memprediksi: {categories[predicted_class]}"
    description_text = descriptions[categories[predicted_class]]
    st.success(result_text)
    st.write(description_text)
```

- `st.title("Prediksi Osteoarthritis pada Lutut")`: Menampilkan judul halaman.
- `st.file_uploader`: Menyediakan widget untuk mengunggah file gambar X-ray.
- Jika tidak ada file yang diunggah, tampilkan teks "Silakan unggah file gambar".
- Jika file diunggah, buka dan tampilkan gambar.
- `with st.spinner('Tunggu sebentar...')`: Menampilkan spinner sementara prediksi sedang diproses.
- `time.sleep(4)`: Menunggu selama 4 detik (simulasi pemrosesan).
- `predictions = import_and_predict(image, model)`: Memanggil fungsi `import_and_predict` untuk mendapatkan prediksi.
- `predicted_class = np.argmax(predictions)`: Mendapatkan kelas dengan probabilitas tertinggi.
- `result_text = f"A.I memprediksi: {categories[predicted_class]}"`: Membuat teks hasil prediksi.
- `description_text = descriptions[categories[predicted_class]]`: Mendapatkan deskripsi sesuai prediksi.
- `st.success(result_text)`: Menampilkan hasil prediksi.
- `st.write(description_text)`: Menampilkan deskripsi prediksi.
