# Fingerprint-Based Blood Group Prediction 🩸🖐️

**Project Name:** Fingerprint Blood Group Prediction System

This project implements **machine learning models to predict a person's blood group using fingerprint features** extracted from fingerprint images. It aims to assist in emergency situations and healthcare systems where quick blood group identification is beneficial.

---

## 🚀 Features

* **Fingerprint feature extraction** using image processing.
* **Machine Learning classification** to predict blood groups (A, B, AB, O).
* **User-friendly interface** for loading fingerprint images and displaying predicted blood group.
* **Portable Python implementation** suitable for Raspberry Pi or low-cost biometric devices.

---

## 📂 Project Structure

* `main.ipynb` : Jupyter notebook containing training, testing, and prediction pipeline.
* `dataset/` : Contains fingerprint images with labeled blood groups.
* `models/` : Saved trained models for fast prediction.
* `utils.py` : Helper functions for feature extraction and preprocessing.

---

## ⚙️ Installation

1️⃣ **Clone the repository:**

```bash
git clone https://github.com/vijaydasp/Fingerprint-BloodGroup-Prediction.git
cd Fingerprint-BloodGroup-Prediction
```

2️⃣ **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Dependencies include:**

* `opencv-python`
* `scikit-learn`
* `numpy`
* `matplotlib`
* `pandas`

---

## ▶️ Usage

Run the Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

or adapt `main.ipynb` logic into your Python pipeline for live fingerprint prediction.

---

## 🩺 How it works

1. **Fingerprint Image Capture:** Capture a clear fingerprint image.
2. **Feature Extraction:** Extract ridge patterns, minutiae, and texture features.
3. **Prediction:** Feed extracted features into the trained ML classifier.
4. **Result:** Display the predicted blood group.

---

## 💡 Applications

✅ Emergency blood group identification during rescue operations.
✅ Integration with biometric attendance and health monitoring systems.
✅ Academic learning on biometric-based health predictions.

---

## 🛠️ Troubleshooting

* Ensure fingerprint images are clear with high contrast.
* If prediction is inaccurate, retrain using your dataset in the notebook.
* Ensure consistent preprocessing of fingerprint images during testing and training.

---

## 🤝 Contributing

Pull requests for model improvement, dataset expansion, and GUI integration are welcome.

---

## 📧 Contact

**Developer:** Vijay Das

**LinkedIn:** [vijaydasp](https://www.linkedin.com/in/vijay-das-p-a42068283?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BxyyRRfIGRJ%2BYk8u1yhtC9g%3D%3D)

---

**Predict blood groups efficiently using your fingerprint with this open-source solution!**
