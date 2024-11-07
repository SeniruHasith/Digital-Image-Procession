import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from flask import Flask, request, render_template

# Set paths relative to the root directory of the project
base_dir = os.path.dirname(os.path.dirname(__file__))  # Root directory of the project
normal_images_path = os.path.join(base_dir, "dataset", "normal_augmented-images")
malignant_images_path = os.path.join(base_dir, "dataset", "malignant")
model_path = os.path.join(base_dir, "Task05", "cancer_detection_model.h5")


# Load and preprocess the images
def load_and_preprocess_images(image_path, label):
    images = []
    labels = []
    for filename in os.listdir(image_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_filepath = os.path.join(image_path, filename)
            image = Image.open(image_filepath).resize((224, 224))  # Resize to 224x224
            image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
            images.append(image)
            labels.append(label)  # Use passed label
    return np.array(images), np.array(labels)


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image


def main():
    # Load images and labels
    try:
        normal_images, normal_labels = load_and_preprocess_images(normal_images_path, label=0)
        malignant_images, malignant_labels = load_and_preprocess_images(malignant_images_path, label=1)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # Concatenate datasets
    all_images = np.concatenate((normal_images, malignant_images), axis=0)
    all_labels = np.concatenate((normal_labels, malignant_labels), axis=0)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    # Check if a GPU is available
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    print(f"Using device: {device}")

    # Check if a pre-trained model exists
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model = load_model(model_path)
    else:
        # Build and train the model
        with tf.device(device):
            model = build_model()

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy',
                          metrics=['accuracy'])

            # Data augmentation
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
                epochs=60,
                validation_data=(X_val, y_val),
                steps_per_epoch=len(X_train) // 32,
                verbose=1
            )

            # Save the trained model
            model.save(model_path)


# Flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['image']
        image = Image.open(file)

        # Preprocess the image
        image = preprocess_image(image)

        # Load the trained model
        model = load_model('cancer_detection_model.h5')

        # Predict the image
        prediction = model.predict(np.expand_dims(image, axis=0))

        # Set result based on prediction threshold
        if prediction[0][0] > 0.5:
            result = "Cancer Detected"
        else:
            result = "No Cancer Detected"

        return render_template('index.html', result=result)

    return render_template('index.html')


if __name__ == '__main__':
    main()
    app.run(debug=True)
