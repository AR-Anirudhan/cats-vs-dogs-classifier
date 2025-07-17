# 🐶🐱 Cats vs Dogs Classifier (Deep Learning with MobileNetV2)

A computer vision project that classifies images of cats and dogs using **Transfer Learning** with **MobileNetV2**, achieving **95% validation accuracy**. Built and trained using **TensorFlow** in **Google Colab**, with real-time prediction support via uploaded images.

---

## 📌 Overview

This project demonstrates how to apply **transfer learning** for image classification using a pre-trained CNN. It uses the **Cats vs Dogs filtered dataset** and applies advanced **data augmentation**, **fine-tuning**, and **threshold-based prediction logic** for clean, real-world outputs.

---

## 🚀 Technologies Used

- Python 3.x  
- Google Colab  
- TensorFlow / Keras  
- MobileNetV2 (pretrained on ImageNet)  
- Matplotlib & NumPy  
- ImageDataGenerator (for augmentation)

---

## 📁 Dataset

- 📦 **Source**: [Cats vs Dogs Filtered Dataset (TensorFlow)](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)
- Automatically downloaded and extracted with:
  ```python
  dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
  zip_path = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=dataset_url)





