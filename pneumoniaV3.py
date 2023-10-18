import numpy as np
import cv2
from tensorflow import keras
from tensorflow import keras as tfk
import tensorflow as tf


# Modeli yüklemek için fonksiyon
def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

def preprocess_image(image_or_path):
    if isinstance(image_or_path, str):  # Eğer bir dosya yolu verilmişse
        img = cv2.imread(image_or_path)
    else:  # Eğer bir NumPy dizisi verilmişse
        img = image_or_path

    if img is None:
        print("Görüntü yüklenemedi.")
        return None

    img = cv2.resize(img, (256, 256))

    if len(img.shape) == 2:  # Eğer görüntü gri tonlamalıysa
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Gri tonlamalıyı BGR'ye çevir

    # Normalizasyon işlemi (0-1 arasına ölçekleme)
    img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img


# Tahmin yapmak için fonksiyon
def make_prediction(model, preprocessed_image):
    prediction = model.predict(preprocessed_image)
    return prediction

# Tahmin sonucunu yorumlamak için fonksiyon
def interpret_result(prediction):
    if prediction >= 0.5:
        return "Pneumonia detected"
    else:
        return "Normal Pneumonia Not Detected"

# Heatmap için fonksiyon
def generate_heatmap(model, preprocessed_image):
    # Modelin son konvolüsyon katmanını al
    last_conv_layer = model.get_layer('mixed8')  # Katmanın adını modelinize göre güncelleyin
    
    # Grad-CAM işlemi için modeli oluştur
    grad_model = tfk.Model([model.inputs], [last_conv_layer.output, model.output])
    
    # Görüntü üzerinde tahmin ve gradyan hesaplama
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(preprocessed_image)
        loss = predictions[:, 0]
        
    # Gradyanı hesapla
    grads = tape.gradient(loss, conv_output)
    
    # Gradyan ve çıktı arasında bir ağırlıklı ortalama hesapla
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    
    # Heatmap'i normalize et
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap[0]

if __name__ == "__main__":
    # Modeli yükle
    model_path = "model path\\model.h5"
    model = load_model(model_path)

    # Modelin beklediği giriş şeklini yazdır
    print("Modelin beklediği giriş şekli:", model.input_shape)
    
    # Örnek bir görüntü yolu (Bu kısmı değiştirebilirsiniz)
    image_path = "image path"

    # Görüntüyü ön işleme
    preprocessed_image = preprocess_image(image_path)

    # Tahmin yap
    prediction = make_prediction(model, preprocessed_image)
    print("Raw Prediction:", prediction)  # Bu satırı ekleyin
    # Sonucu yorumla
    result = interpret_result(prediction)
    print(result)
