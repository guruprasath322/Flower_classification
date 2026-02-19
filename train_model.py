import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Define constants
DATA_DIR = "flowers"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 10 # You can adjust this

def train_and_save_model():
    print("Setting up data generators...")
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED
    )

    val_data = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED
    )

    num_classes = len(train_data.class_indices)
    print("Class Indices:", train_data.class_indices)
    print("Num classes:", num_classes)

    print("Building model...")
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"Training model for {EPOCHS} epochs...")
    history = model.fit(
        train_data,
        steps_per_epoch=train_data.samples // BATCH_SIZE,
        validation_data=val_data,
        validation_steps=val_data.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    print("Saving model to flower_model.keras...")
    model.save("flower_model.keras")
    print("Model saved successfully!")

if __name__ == "__main__":
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found anywhere.")
    else:
        train_and_save_model()
