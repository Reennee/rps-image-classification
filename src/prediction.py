import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.optimizers import Adam

# Class labels (should match the order in your dataset)
CLASS_NAMES = ['paper', 'rock', 'scissors']

def load_model(model_path='models/best_model.h5'):
    """
    Load the trained Keras model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(img_path, target_size=(150, 150)):
    """
    Load and preprocess the image to match model input.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(img_path, model_path='models/best_model.h5', target_size=(150, 150)):
    """
    Predict the class of a given image using the trained model.
    Returns the predicted class and confidence.
    """
    model = load_model(model_path)
    processed_image = preprocess_image(img_path, target_size=target_size)
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions[0]))
    return predicted_class, confidence

def retrain_model(data_dir="data/train", model_path="models/best_model.h5", epochs=3):
    """
    Retrain the existing model on new data from a given directory.
    The directory should have the format: class_name/image.jpg
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = keras_load_model(model_path)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        subset='validation'
    )

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_data, validation_data=val_data, epochs=epochs)

    model.save(model_path)
    return model