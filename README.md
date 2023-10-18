🩺 Pneumonia Detection from X-Ray Images 🩺
Overview
This project uses a deep learning model to detect pneumonia from chest X-ray images. The model is deployed as a web application using Streamlit. Users can upload an X-ray image, and the model will classify it as either "Pneumonia detected" or "Normal Pneumonia Not Detected."

Features
Upload X-ray images in JPG, JPEG, or PNG formats.
Real-time pneumonia detection.
Prediction heatmap to visualize the affected areas.
Model accuracy: 96%

Tech Stack
- Python
- TensorFlow/Keras
- OpenCV
- Streamlit
- Matplotlib
  
Installation
1- Clone the repository:
git clone https://github.com/doruq-IT/Pneumonia-Detection-from-X-Ray-Images.git

2- Navigate to the project directory:
cd Pneumonia-Detection-from-X-Ray-Images

3- Install the required packages:
pip install -r requirements.txt

4- Run the Streamlit app:
streamlit run streamlit_V1.py

Usage
Open the Streamlit app in your web browser.
Upload a chest X-ray image using the file uploader.
Wait for the model to classify the image.
View the prediction result and heatmap.
Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License
MIT


