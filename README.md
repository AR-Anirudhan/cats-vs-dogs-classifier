🐶🐱 Cats vs Dogs Classifier (Deep Learning with MobileNetV2)

A computer vision project that classifies images of cats and dogs using **Transfer Learning** with **MobileNetV2**, achieving **95% validation accuracy**. Built and trained using **TensorFlow** in **Google Colab**, with real-time prediction support via uploaded images.



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
````

* Contains `train/` and `validation/` directories with labeled images  
* Preprocessed to `150x150` resolution

---

## 🧠 Model Architecture

* ✅ **Base**: MobileNetV2 (frozen initially, then fine-tuned)
* ✅ **Custom Head**:
  * GlobalAveragePooling2D  
  * Dense(512, ReLU)  
  * Dropout(0.3)  
  * Dense(1, Sigmoid)
* ✅ **Loss**: Binary Crossentropy  
* ✅ **Optimizer**: Adam (lower LR during fine-tuning)  
* ✅ **Metrics**: Accuracy

---

## 📊 Results

* 🟢 **Validation Accuracy**: ~95%
* 📈 Two-phase training:
  * Phase 1: Base frozen  
  * Phase 2: Fine-tuning last 20 layers
* 🧪 Visualized predictions on uploaded images

---

## 🎯 Key Features

* ✅ Data augmentation (rotation, zoom, shift, flip)  
* ✅ Real-time image classification using Colab uploads  
* ✅ Confidence thresholds for improved decision boundaries  
* ✅ Matplotlib-based result display with prediction label

---

## 📸 Sample Prediction Output

![Prediction Output](https://github.com/user-attachments/assets/5e0a4ca3-48a1-4553-9c4d-e1f903b8b4ee)





