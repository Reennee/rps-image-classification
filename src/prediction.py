import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.optimizers import Adam

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

def retrain_model(new_data_dir, model_path='models/best_model.h5', epochs=2):
    model = load_model(model_path)
    # Freeze all layers except the last Dense
    for layer in model.layers[:-1]:
        layer.trainable = False

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        new_data_dir, target_size=(150, 150), batch_size=16, class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        new_data_dir, target_size=(150, 150), batch_size=16, class_mode='categorical', subset='validation'
    )

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save(model_path)
    np.save('models/history.npy', history.history)
    return model
