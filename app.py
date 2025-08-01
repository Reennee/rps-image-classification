import streamlit as st
from PIL import Image
import numpy as np
from src.prediction import predict_image, load_model
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import requests

st.title("Rock-Paper-Scissor cycle")

# Model health check
try:
    _ = load_model()
    st.success("Model Status: Online")
except Exception as e:
    st.error(f"Model Status: Offline ({e})")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Save uploaded file temporarily
    temp_path = "temp_uploaded_image.jpg"
    image_pil.save(temp_path)

    # Prediction
    label, confidence = predict_image(temp_path)
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    os.remove(temp_path)

# --- Bulk Upload ---
st.header("Bulk Upload & Retrain Model")
with st.form("bulk_upload_form"):
    st.write("Upload multiple images to add to the training set and retrain the model.")
    bulk_files = st.file_uploader("Select images to upload (multiple allowed)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    class_label = st.selectbox("Select class label for these images", ["rock", "paper", "scissors"])
    retrain_button = st.form_submit_button("Upload & Retrain")

    if retrain_button and bulk_files:
        with st.spinner("Uploading images and retraining model..."):
            success_count = 0
            for file in bulk_files:
                files = {"file": (file.name, file.getvalue(), file.type)}
                data = {"label": class_label}
                try:
                    response = requests.post("http://localhost:8000/retrain", files=files, data=data)
                    if response.status_code == 200:
                        success_count += 1
                    else:
                        st.warning(f"Failed to upload {file.name}: {response.text}")
                except Exception as e:
                    st.warning(f"Error uploading {file.name}: {e}")
            if success_count:
                st.success(f"Uploaded and retrained with {success_count} images for class '{class_label}'.")
            else:
                st.error("No images were successfully uploaded for retraining.")

# --- Visualizations Section ---
with st.expander("Show Data & Model Visualizations"):
    st.header("Class Distribution (Train Set)")
    train_dir = "data/train"
    class_counts = {}
    class_samples = {}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            images = [img for img in os.listdir(class_path) if img.lower().endswith((".jpg", ".jpeg", ".png"))]
            class_counts[class_name] = len(images)
            class_samples[class_name] = random.sample(images, min(3, len(images))) if images else []
    # Bar chart
    st.bar_chart(class_counts)

    st.subheader("Sample Images per Class")
    cols = st.columns(len(class_samples))
    for idx, (class_name, samples) in enumerate(class_samples.items()):
        with cols[idx]:
            st.markdown(f"**{class_name.capitalize()}**")
            for img_name in samples:
                img_path = os.path.join(train_dir, class_name, img_name)
                st.image(img_path, width=100)

    st.header("Model Evaluation (Test Set)")
    # Preparing test generator
    test_dir = "data/test"
    img_size = (150, 150)
    batch_size = 32
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    # Loading model
    model = keras_load_model('models/best_model.h5')
    # Prediction
    Y_pred = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    st.text(report)

    # Accuracy/Loss Curves
    st.subheader("Accuracy and Loss Curves")
    # Loading history
    history_path = 'models/history.npy'
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history['accuracy'], label='Train Accuracy')
        ax1.plot(history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Accuracy over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax2.plot(history['loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Val Loss')
        ax2.set_title('Loss over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        st.pyplot(fig_hist)
    else:
        st.info("Training history not found. Please save training history as 'models/history.npy' to display curves.")