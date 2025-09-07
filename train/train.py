import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_loader import DataLoader


def create_barbell_model(input_size=(256, 256, 3)):
    """Create MobileNetV3-Small based barbell detector"""

    # Base model
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_size,
        include_top=False,
        weights="imagenet",
        alpha=1.0,  # Width multiplier for smaller model
        minimalistic=True,  # Simplified architecture
    )

    # Freeze base model initially for transfer learning
    base_model.trainable = True

    # Custom detection head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Output layers
    bbox_output = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox")(x)
    confidence_output = tf.keras.layers.Dense(
        1, activation="sigmoid", name="confidence"
    )(x)

    model = tf.keras.Model(
        inputs=base_model.input, outputs=[bbox_output, confidence_output]
    )

    return model


def bbox_loss(y_true, y_pred):
    """IoU-based loss for bounding box regression"""
    # Only calculate loss for positive samples (confidence > 0)
    return tf.keras.losses.Huber(delta=0.1)(y_true, y_pred)


def confidence_loss(y_true, y_pred):
    """Focal loss for confidence to handle class imbalance"""
    alpha = 0.25
    gamma = 2.0

    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(
        y_true, y_pred
    )
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_loss = alpha * tf.pow(1 - pt, gamma) * bce

    return tf.reduce_mean(focal_loss)


def train_barbell_detector(dataset_path, epochs=80, batch_size=32, learning_rate=0.001):
    """Main training function"""

    print("Loading dataset...")
    data_loader = DataLoader(dataset_path, input_size=(256, 256))

    # Load training and validation data
    X_train, bbox_train, conf_train = data_loader.load_dataset("train")
    X_val, bbox_val, conf_val = data_loader.load_dataset("val")

    # Check if we have any data
    if len(X_train) == 0:
        raise ValueError(
            "No training data found! Please check your dataset path and structure."
        )
    if len(X_val) == 0:
        raise ValueError(
            "No validation data found! Please check your dataset path and structure."
        )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Positive training samples: {np.sum(conf_train)}")
    print(f"Positive validation samples: {np.sum(conf_val)}")

    # Adjust batch size if necessary
    if batch_size > len(X_train):
        print(f"Warning: batch_size ({batch_size}) > training samples ({len(X_train)})")
        batch_size = max(1, len(X_train) // 4)
        print(f"Adjusted batch_size to: {batch_size}")

    # Create model
    print("Creating model...")
    model = create_barbell_model()

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"bbox": bbox_loss, "confidence": confidence_loss},
        loss_weights={"bbox": 1.0, "confidence": 3.0},  # Higher weight for confidence
        metrics={"confidence": ["accuracy", "precision", "recall"]},
    )

    print("Model summary:")
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=8, factor=0.5, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger("train/output/training_log.csv"),
    ]

    print("Starting training...")
    history = model.fit(
        X_train,
        [bbox_train, conf_train],
        validation_data=(X_val, [bbox_val, conf_val]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # Save final model
    model.save("model/barbell_detector_model.h5")
    print("Training completed!")

    return model, history


def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history["loss"], label="Training Loss")
    axes[0, 0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0, 0].set_title("Model Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Confidence accuracy
    axes[0, 1].plot(history.history["confidence_accuracy"], label="Training Accuracy")
    axes[0, 1].plot(
        history.history["val_confidence_accuracy"], label="Validation Accuracy"
    )
    axes[0, 1].set_title("Confidence Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    # BBox loss
    axes[1, 0].plot(history.history["bbox_loss"], label="Training BBox Loss")
    axes[1, 0].plot(history.history["val_bbox_loss"], label="Validation BBox Loss")
    axes[1, 0].set_title("Bounding Box Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    # Confidence loss
    axes[1, 1].plot(
        history.history["confidence_loss"], label="Training Confidence Loss"
    )
    axes[1, 1].plot(
        history.history["val_confidence_loss"], label="Validation Confidence Loss"
    )
    axes[1, 1].set_title("Confidence Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("train/output/training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


# Usage example
if __name__ == "__main__":
    # Set dataset path
    DATASET_PATH = "./dataset"

    # Training configuration
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

                # Train the model
                model, history = train_barbell_detector(
                    dataset_path=DATASET_PATH,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATE,
                )

        except RuntimeError as e:
            print("Error occurred:", e)

    # Plot training history
    plot_training_history(history)

    print("Training completed!")
    print("Best model saved as: best_barbell_detector.h5")
    print("Final model saved as: barbell_detector_final.h5")
