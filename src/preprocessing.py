import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
def load_preprocess_data(img_size=(150, 150), batch_size=32):
    # Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # No augmentation for testing
    test_datagen = ImageDataGenerator(rescale=1./255)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(project_root, 'data', 'train')
    test_dir = os.path.join(project_root, 'data', 'test')

    # Load training images
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load testing images
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator
