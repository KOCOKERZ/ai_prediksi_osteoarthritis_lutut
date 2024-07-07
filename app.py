# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt  # Tambahkan impor ini
import time

# hide deprecation warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Prediksi Osteoarthritis pada Lutut",
    page_icon=":bone:",
    layout="centered",
    initial_sidebar_state='auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1fcdlh1 {padding: 2rem 1rem 10rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

def import_and_predict(image_data, model):
    img_size = 128
    img = ImageOps.grayscale(image_data)
    img = img.resize((img_size, img_size))
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'model.h5'  # Sesuaikan path ini sesuai kebutuhan
    return tf.keras.models.load_model(model_path)

model = load_model()

# Define the classes and their descriptions
categories = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']
descriptions = {
    'Normal': 'Lutut tampak normal tanpa tanda-tanda osteoartritis.',
    'Doubtful': 'Ada tanda-tanda minor yang mungkin menunjukkan tahap awal osteoartritis.',
    'Mild': 'Ada tanda-tanda ringan osteoartritis, dengan beberapa penyempitan ruang sendi.',
    'Moderate': 'Ada tanda-tanda sedang osteoartritis, dengan penyempitan ruang sendi yang terlihat dan kemungkinan adanya taji tulang.',
    'Severe': 'Ada tanda-tanda parah osteoartritis, dengan penyempitan ruang sendi yang signifikan dan taji tulang yang besar.'
}

# Title and description
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
        time.sleep(5)  # Simulate a delay
        predictions = import_and_predict(image, model)
        predicted_class = np.argmax(predictions)

    result_text = f"A.I memprediksi: {categories[predicted_class]}"
    description_text = descriptions[categories[predicted_class]]
    st.success(result_text)
    st.write(description_text)
