import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pneumoniaV3 import load_model, preprocess_image, make_prediction, interpret_result, generate_heatmap

st.set_option('deprecation.showPyplotGlobalUse', False)

# Modeli yükle
model_path = "C:\\Users\\okan_\\Desktop\\alexandre\\rontgen-hastalik\\model.h5"
model = load_model(model_path)

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This tool uses a deep learning model to detect pneumonia from chest X-ray images.")
st.sidebar.write("Model Accuracy: 96%")

# Main Content
st.title("🩺 Pneumonia Detection from X-Ray Images 🩺")
st.write("## This tool uses a deep learning model to detect pneumonia.")
st.write("Please upload a chest X-ray image for pneumonia classification.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("### Uploaded X-ray Image:")
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    with st.spinner('Classifying...'):
        # PIL görüntüsünü NumPy dizisine dönüştür
        image = np.array(image)
        
        # Görüntüyü ön işleme ve tahmin yap
        preprocessed_image = preprocess_image(image)
        
        # Heatmap oluştur
        heatmap = generate_heatmap(model, preprocessed_image)
        
        # Heatmap'i görüntüle
        plt.matshow(heatmap)
        plt.title('Prediction Heatmap')
        st.pyplot()
        
        if preprocessed_image is not None:
            prediction = make_prediction(model, preprocessed_image)
            
            # Sonucu yorumla ve göster
            result = interpret_result(prediction)
            st.write("Result:", result)
            
            if result == "Pneumonia detected":
                st.error("Result: Pneumonia detected 😷")
            else:
                st.success("Result: Normal Pneumonia Not Detected 🥳")
        
        # fig, ax = plt.subplots()
        # ax.matshow(heatmap)
        # plt.title('Prediction Heatmap')
        # st.pyplot(fig)

