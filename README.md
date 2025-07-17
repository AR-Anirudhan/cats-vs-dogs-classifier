# 🐶🐱 Cats vs Dogs Classifier (Deep Learning with MobileNetV2)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AR-Anirudhan/cats-vs-dogs-classifier/blob/main/cats%20vs%20dogs.ipynb)


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

* Contains `train/` and `validation/` directories with labeled images  
* Preprocessed to `150x150` resolution

---

## 🧠 Model Architecture

* ✅ **Base Model**: MobileNetV2 (frozen initially, then fine-tuned)

* ✅ **Custom Head**:
  * GlobalAveragePooling2D  
  * Dense layer with 512 units (ReLU activation)  
  * Dropout layer (rate = 0.3)  
  * Output Dense layer (1 unit, Sigmoid activation)

* ✅ **Loss Function**: Binary Crossentropy  
* ✅ **Optimizer**: Adam (with reduced learning rate for fine-tuning)  
* ✅ **Metric**: Accuracy

---

## 📊 Results

* 🟢 **Validation Accuracy**: ~95%

* 📈 **Two-phase training**:
  * Phase 1: Base model frozen  
  * Phase 2: Fine-tuned last 20 layers of MobileNetV2

* 🧪 Visualized predictions on uploaded images with confidence scores

---

## 🎯 Key Features

* ✅ Data augmentation (rotation, zoom, shift, horizontal flip)  
* ✅ Real-time prediction using uploaded images in Google Colab  
* ✅ Threshold-based decision system for confident predictions  
* ✅ Visual output using Matplotlib with prediction labels

---

## 📸 Sample Prediction Output

**Uploaded file**: `dog.jpg`  
**Prediction**: 🐶 Dog  
**Confidence**: `0.92`

**Threshold Logic**:
- `> 0.8` → 🐶 Dog  
- `< 0.2` → 🐱 Cat  
- `0.2 – 0.8` → ❓ Not Sure

---

## ▶️ How to Use (Colab Notebook)

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. The dataset will be auto-downloaded
3. Train the model using the training and validation generators
4. Upload an image to make a prediction
5. View the result as an image with label and confidence score

---

## 📚 Learning Outcomes

* Transfer learning with pretrained CNNs  
* Fine-tuning deep models with selective unfreezing  
* Real-time inference with preprocessing pipelines  
* Building robust image classifiers using augmentation

---

## 📬 Contact

**Anirudhan A R**  
🎓 3rd Year B.Tech in Artificial Intelligence & Data Science  
🧠 Specialization: Cloud and Edge Computing  
🏫 KL University

- 📫 Email: [anirudhanksr59@gmail.com](mailto:anirudhanksr59@gmail.com)  
- 🔗 LinkedIn: [linkedin.com/in/anirudhan-ar-36b6472b4](https://www.linkedin.com/in/anirudhan-ar-36b6472b4)  
- 💻 GitHub: [github.com/AR-Anirudhan](https://github.com/AR-Anirudhan)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

* [TensorFlow](https://www.tensorflow.org/)  
* [Google Colab](https://colab.research.google.com/)  
* [Cats vs Dogs Dataset (Filtered)](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)






