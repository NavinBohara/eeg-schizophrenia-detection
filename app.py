import os
import numpy as np
import zipfile
import random
import mne
import torch
import torch.nn as nn
import transformers
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Fix randomness
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("🧠 SchizoScan AI – AI-Powered Insights for Mental Wellness.")
st.sidebar.header("Upload EEG Dataset")

uploaded_file = st.sidebar.file_uploader("Upload EEG ZIP file", type=["zip"])

if uploaded_file:
    extract_folder = "extracted_eeg"
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    st.sidebar.success("✅ Files Extracted Successfully!")

    def load_eeg(file_path):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        data, _ = raw.get_data(return_times=True)
        return data

    def extract_features(data):
        mean_vals = np.mean(data, axis=1)
        var_vals = np.var(data, axis=1)
        psd_vals = np.log(np.abs(np.fft.fft(data, axis=1)) ** 2)
        entropy = -np.sum((psd_vals / np.sum(psd_vals, axis=1, keepdims=True)) * np.log(psd_vals + 1e-10), axis=1)
        return np.concatenate((mean_vals, var_vals, entropy), axis=0)

    file_paths = [os.path.join(extract_folder, file) for file in os.listdir(extract_folder) if file.endswith(".edf")]
    
    if len(file_paths) == 0:
        st.error("❌ No .edf files found in the uploaded ZIP.")
        st.stop()

    X, y = [], []
    for file in file_paths:
        try:
            data = load_eeg(file)
            features = extract_features(data)
            X.append(features)
            # Improved logic: classify based on 'schizo' or 'healthy'
            file_name = file.lower()
            if "schizo" in file_name:
                y.append(1)
            elif "healthy" in file_name or "h" in file_name:
                y.append(0)
            else:
                st.warning(f"⚠️ File '{file}' doesn't follow naming convention. Skipped.")
        except Exception as e:
            st.warning(f"⚠️ Could not process {file}: {str(e)}")

    if len(X) < 4 or len(set(y)) < 2:
        st.error("❌ Not enough labeled EEG samples or only one class present. Please upload a valid dataset with both classes.")
        st.stop()

    X, y = np.array(X), np.array(y)

    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    st.write("🧾 Class Distribution:", dict(zip(unique, counts)))

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Try stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        st.warning("⚠️ Stratified split failed. Falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def augment_data(X, y, num_samples=5):
        X_aug, y_aug = [], []
        for _ in range(num_samples):
            noise = np.random.normal(0, 0.05, X.shape)
            X_aug.append(X + noise)
            y_aug.append(y)
        return np.vstack(X_aug), np.hstack(y_aug)

    X_train, y_train = augment_data(X_train, y_train)

    st.write("🟡 **Training Model... (BERT Feature Extraction in Progress)**")

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    class BERTFeatureExtractor(nn.Module):
        def __init__(self, hidden_dim=128):
            super().__init__()
            self.bert = transformers.AutoModel.from_pretrained("bert-base-uncased").to(device)
            self.fc = nn.Linear(768, hidden_dim).to(device)

        def forward(self, x):
            text_data = [" ".join(map(str, row[:10])) for row in x.tolist()]
            inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.bert(**inputs).last_hidden_state.mean(dim=1)
            return self.fc(outputs).cpu().detach().numpy()

    bert_model = BERTFeatureExtractor().eval()
    X_train_bert = bert_model(torch.tensor(X_train, dtype=torch.float32))
    X_test_bert = bert_model(torch.tensor(X_test, dtype=torch.float32))

    st.write("🟡 **Training XGBoost Model...**")
    xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_clf.fit(X_train_bert, y_train)

    y_pred = xgb_clf.predict(X_test_bert)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ **Model Accuracy:** {accuracy:.4f}")

    # Classification report
    class_report = classification_report(y_test, y_pred, target_names=["Healthy", "Schizophrenia"])
    st.write("📊 **Classification Report:**")
    st.text(class_report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Schizophrenia"])
    disp.plot(cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Bar graph of label counts
    healthy_count = np.sum(y == 0)
    schizophrenia_count = np.sum(y == 1)

    fig, ax = plt.subplots()
    bars = ax.bar(["Healthy", "Schizophrenia"], [healthy_count, schizophrenia_count], color=["blue", "red"])
    ax.set_xlabel("Condition")
    ax.set_ylabel("Count")
    ax.set_title("Dataset Distribution")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    st.pyplot(fig)

    # Upload for single prediction
    st.sidebar.header("Upload EEG File for Prediction")
    pred_file = st.sidebar.file_uploader("Upload a single EEG file (.edf)", type=["edf"])

    if pred_file:
        temp_file = "temp.edf"
        with open(temp_file, "wb") as f:
            f.write(pred_file.getbuffer())

        pred_data = load_eeg(temp_file)
        pred_features = extract_features(pred_data)
        pred_features = scaler.transform([pred_features])
        pred_features_bert = bert_model(torch.tensor(pred_features, dtype=torch.float32))

        pred_label = xgb_clf.predict(pred_features_bert)[0]
        diagnosis = "Healthy" if pred_label == 0 else "Schizophrenia"
        st.sidebar.success(f"🧠 **Prediction:** {diagnosis}")

        st.subheader("🧠 Final Diagnosis Result")
        if pred_label == 0:
            st.success("✅ The EEG pattern suggests a **Healthy** brain.")
        else:
            st.error("⚠️ The EEG pattern indicates **Schizophrenia**.")
