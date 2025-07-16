import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

st.title("♻️ AI-Based Waste Classifier")
st.write("Upload an image of waste")

# Paths
dataset_path = "waste_dataset"
model_path = "waste_model.h5"

# Load or train model
if not os.path.exists(model_path):
    st.warning("⚠️ Training the model... This might take a while!")

    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, epochs=10, validation_data=val_data)
    model.save(model_path)
    st.success("✅ Model trained and saved!")
else:
    model = tf.keras.models.load_model(model_path)

# Get class labels (this is important to fix the "val" issue)
datagen_temp = ImageDataGenerator(rescale=1.0/255)
temp_data = datagen_temp.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=1,
    class_mode="categorical"
)
class_names = list(temp_data.class_indices.keys())

# Image Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_resized = image.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Predicted Waste Category: **{predicted_class.capitalize()}**")
