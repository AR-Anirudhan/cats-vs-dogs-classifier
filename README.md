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
  * Dense(512, relu)
  * Dropout(0.3)
  * Dense(1, sigmoid)
* ✅ **Loss**: Binary Crossentropy
* ✅ **Optimizer**: Adam (lower LR during fine-tuning)
* ✅ **Metrics**: Accuracy

---

## 📊 Results

* 🟢 **Validation Accuracy**: \~95%
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

```bash
Uploaded file: image.jpg
Prediction: 🐶 Dog (Confidence: 0.92)
```

Threshold logic:

* `> 0.8` → Dog
* `< 0.2` → Cat
* `0.2–0.8` → Not Sure ❓

---

## ▶️ How to Use (Colab Notebook)

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Dataset is auto-downloaded
3. Train the model using the training and validation generators
4. Upload an image to make a prediction
5. View result as an image with label and confidence

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

* 📫 Email: [anirudhanksr59@gmail.com](mailto:anirudhanksr59@gmail.com)
* 🔗 LinkedIn: [linkedin.com/in/anirudhan-ar-36b6472b4](https://www.linkedin.com/in/anirudhan-ar-36b6472b4)
* 💻 GitHub: [github.com/AR-Anirudhan](https://github.com/AR-Anirudhan)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

* [TensorFlow](https://www.tensorflow.org/)
* [Google Colab](https://colab.research.google.com/)
* [Cats vs Dogs Dataset (Filtered)](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)

```

---



