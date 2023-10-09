import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
st.markdown(
    """
    <style>
    .stApp {
        background-color: #83e627;  /* Koyu sarı renk */
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit uygulamasını başlat
st.markdown('<div style="display: flex; justify-content: flex-end; margin-top:-70px"><img src="https://i.pinimg.com/originals/4a/73/1f/4a731f6a5480f6ee8b9bfb34168c333b.gif" alt="GIF" width="100%" style="max-width: 400px; margin-right: 160px;"></div>', unsafe_allow_html=True)
st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 30px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.1);">🌻Flower Forecast App🌻</p>', unsafe_allow_html=True)
st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">💐Flower Species💐</p>', unsafe_allow_html=True)
st.image("turler.png", use_column_width=True)
# Kullanıcıdan resim yükleme yöntemini seçmesini isteyin
st.sidebar.title("Image Upload Method")
upload_method = st.sidebar.radio("Please select image upload type:", ["Install from your computer", "Install with Internet Connection"])

uploaded_image = None  # Kullanıcının yüklediği resmi saklamak için

if upload_method == "Install from your computer":
    # Kullanıcıdan resim yükleme
    #st.write("Lütfen bir çiçek resmi yükleyin:")
    uploaded_image = st.file_uploader("Please upload a flower picture:", type=["jpg", "png", "jpeg"])
elif upload_method == "Install with Internet Connection":
    # Kullanıcıdan internet linki alın
    st.write("Install with Internet Connection:")
    image_url = st.text_input("Image Link")

# Model seçimi
st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio("Please select a model:", ["CNN_model", "VGG16_model", "ResNet_model", "Xception_model", "NASNetMobile_model"])


# Resmi yükle ve tahmin et butonları
if uploaded_image is not None or (upload_method == "Install with Internet Connection" and image_url):
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌼Image of your choice🌼</p>', unsafe_allow_html=True)
    #st.write("Seçtiğiniz Resim")
    if uploaded_image is not None:
        st.image(uploaded_image, caption='', use_column_width=True)
    elif upload_method == "Install with Internet Connection" and image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='', use_column_width=True)
        except Exception as e:
            st.error("There was an error uploading the image. Please enter a valid internet link.")

# Model bilgisi düğmesi
if st.sidebar.button("Information about the model"):
    st.markdown(f'<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌷{selected_model}🌷</p>', unsafe_allow_html=True)

    if selected_model == "CNN_model":
        st.write("CNN_model is a basic Convolutional Neural Network (CNN) model. It contains convolutional layers, pooling layers and fully connected layers. It is typically used for basic visual classification tasks.")
    elif selected_model == "VGG16_model":
        st.write("VGG16_model is a 16-layer deep Convolutional Neural Network model. It contains alternating convolutional and pooling layers. It is used for tasks such as visual classification and object recognition.")
    elif selected_model == "ResNet_model":
        st.write("ResNet_model is a deep Convolutional Neural Network model that uses 'residual' blocks to make it easier to train deep networks. It is used to improve the training of deep networks.")
    elif selected_model == "Xception_model":
        st.write("Xception Model: Xception is a model that fundamentally changes the convolutional neural network architecture. It efficiently extracts features and can be used for classification tasks.")
    elif selected_model == "NASNetMobile_model":
        st.write("NASNetMobile is a model developed with automated architecture search and optimized specifically for lightweight and mobile devices. It can be used for transfer learning for mobile applications and portable devices.")
   
                
# Tahmin yap butonu
if st.button("Predict"):
    if upload_method == "Install from your computer" and uploaded_image is not None:
        image = Image.open(uploaded_image)
    elif upload_method == "Install with Internet Connection" and image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error("There was an error uploading the image. Please enter a valid internet link.")

    # Kullanıcının seçtiği modele göre modeli yükle
    if selected_model == "CNN_model":
        model_path = 'CNN_model.h5'
    elif selected_model == "VGG16_model":
        model_path = 'VGG16.h5'
    elif selected_model == "ResNet_model":
        model_path = 'Resnet50.h5'
    elif selected_model == "Xception_model":
        model_path = 'Xception.h5'
    elif selected_model == "NASNetMobile_model":
        model_path = 'NASNetMobile.h5'

    # Seçilen modeli yükle
    model = tf.keras.models.load_model(model_path, compile=False)   # , compile=False

    # Resmi model için hazırla ve tahmin yap
    if 'image' in locals():
        image = image.resize((224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Tahmin yap
        prediction = model.predict(image)

        # Tahmin sonuçlarını göster
        class_names = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]  # Modelin tahmin sınıfları
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        st.markdown(f'<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌷{selected_model} Forecasting🌷</p>', unsafe_allow_html=True)

        st.info(f"Forecast Result: {predicted_class}")
        st.info(f"Forecast Confidence: {confidence:.2f}")
        
        st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">📊 Forecast Probabilities 📊</p>', unsafe_allow_html=True)
        prediction_df = pd.DataFrame({'Flower Types': class_names, 'Possibilities': prediction[0]})
        st.bar_chart(prediction_df.set_index('Flower Types'))
