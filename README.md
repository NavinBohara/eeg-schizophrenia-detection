# 🧠 SchizoScan AI – AI-Powered EEG-based Schizophrenia Detection

**SchizoScan AI** is an intelligent mental wellness diagnostic tool that uses deep learning and machine learning to detect signs of schizophrenia from EEG signals. The app extracts features from `.edf` EEG files, uses BERT for feature extraction, and classifies signals using an XGBoost model — all in a clean and interactive Streamlit interface.

---

## 🚀 Features

- Upload and analyze `.edf` EEG files in ZIP format
- Feature extraction (mean, variance, entropy, PSD)
- BERT-based feature transformation (using `bert-base-uncased`)
- Classification using XGBoost
- Live model accuracy and classification report
- Confusion matrix and visual dataset distribution
- Scatter plot visualization with decision boundary
- Predict condition (Healthy or Schizophrenia) on new EEG files

---

## 📁 Folder Structure

Perfect! Here's a complete, professional-quality `README.md` file you can copy and paste into your project directory:

---

```markdown
# 🧠 SchizoScan AI – AI-Powered EEG-based Schizophrenia Detection

**SchizoScan AI** is an intelligent mental wellness diagnostic tool that uses deep learning and machine learning to detect signs of schizophrenia from EEG signals. The app extracts features from `.edf` EEG files, uses BERT for feature extraction, and classifies signals using an XGBoost model — all in a clean and interactive Streamlit interface.

---

## 🚀 Features

- Upload and analyze `.edf` EEG files in ZIP format
- Feature extraction (mean, variance, entropy, PSD)
- BERT-based feature transformation (using `bert-base-uncased`)
- Classification using XGBoost
- Live model accuracy and classification report
- Confusion matrix and visual dataset distribution
- Scatter plot visualization with decision boundary
- Predict condition (Healthy or Schizophrenia) on new EEG files

---

## 📁 Folder Structure

```

project/
│
├── app.py                  # Streamlit application code
├── extracted\_eeg/          # Extracted .edf EEG files (after upload)
├── temp.edf                # Temp file for prediction
├── README.md               # This file
└── requirements.txt        # Python dependencies

````

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/schizoscan-ai.git
cd schizoscan-ai
````

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📦 Requirements

List of major dependencies (included in `requirements.txt`):

* `streamlit`
* `mne`
* `transformers`
* `xgboost`
* `torch`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `pandas`

---

## 📊 Sample Visuals

* Confusion Matrix
* Classification Report
* EEG Feature Distribution (Bar Chart)
* Scatter Plot with Decision Boundary

---

## 📤 Input Format

### EEG ZIP Upload:

* A `.zip` file containing one or more `.edf` files.
* Files should be named to include `h` for healthy or anything else for schizophrenia (e.g., `s1_h.edf` or `s2_s.edf`).

### Single EEG Prediction:

* One `.edf` file uploaded through the sidebar after model training.

---

## 🧪 How It Works

1. EEG data is extracted using `mne`.
2. Features are calculated: mean, variance, entropy from PSD.
3. BERT is used to convert numeric EEG features into embeddings.
4. XGBoost classifier is trained and evaluated.
5. The app predicts schizophrenia or healthy condition from unseen EEG data.

---

## 🔒 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Riddhi \[Your Full Name Optional]**

* GitHub: https://github.com/NavinBohara
* Email: navinbohara5@gmail.com

---

## 🙌 Acknowledgements

* [MNE-Python](https://mne.tools/stable/index.html) for EEG data handling
* [Hugging Face Transformers](https://huggingface.co/transformers/) for BERT
* [Streamlit](https://streamlit.io/) for app deployment
* EEG dataset used from [physionet](https://physionet.org/) or \[Kaggle] (add link if used)

---

```

Let me know if you'd like me to create the `requirements.txt` or deployment instructions (Streamlit sharing, Hugging Face Spaces, or AWS).
```
