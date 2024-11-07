import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Set paths to the image datasets
normal_images_path = "D:\\Projects\\Image-processing-project\\dataset\\normal_augmented-images"
malignant_images_path = "D:\\Projects\\Image-processing-project\\dataset\\malignant"

# Load and preprocess the images
def load_and_preprocess_images(image_path):
    images = []
    labels = []
    for filename in os.listdir(image_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_filepath = os.path.join(image_path, filename)
            image = Image.open(image_filepath)
            image = image.resize((224, 224))  # Resize to 224x224
            image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
            images.append(image)
            if "normal" in filename.lower():
                labels.append(0)  # Normal image
            else:
                labels.append(1)  # Malignant image
    return np.array(images), np.array(labels)

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def main():
    try:
        normal_images, normal_labels = load_and_preprocess_images(normal_images_path)
        malignant_images, malignant_labels = load_and_preprocess_images(malignant_images_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    all_images = np.concatenate((normal_images, malignant_images), axis=0)
    all_labels = np.concatenate((normal_labels, malignant_labels), axis=0)

    X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    # Check if a GPU is available
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    print(f"Using device: {device}")

    # Build the model
    with tf.device(device):
        model = build_model()

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=50,
            validation_data=(X_val, y_val),
            steps_per_epoch=len(X_train) // 32,
            verbose=1
        )

    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.2f}")
    print(f"Validation Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()