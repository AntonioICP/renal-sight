# ------------------------ IMPORTS ------------------------
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, Input, layers, optimizers
from keras.applications import VGG16, ResNet50, EfficientNetB0

# --------------------- CONFIGURATION ---------------------
IMG_SIZE = (224, 224)

# --------------------- BASELINE CNN ----------------------
def build_baseline_model():
    model = Sequential()

    model.add(Input((224, 224, 3)))

    model.add(layers.Rescaling(1./255))

    model.add(layers.Conv2D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )

    return model

# ------------------ PRETRAINED MODELS --------------------
def set_nontrainable_layers(model):
    model.trainable = False
    return model

def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    # $CHALLENGIFY_BEGIN
    base_model = set_nontrainable_layers(model)
    pooling_layer = layers.GlobalAveragePooling2D()
    dense_layer = layers.Dense(256, activation='relu')
    prediction_layer = layers.Dense(1, activation='sigmoid')


    model = Sequential([
        base_model,
        pooling_layer,
        dense_layer,
        prediction_layer
    ])
    return model

def build_pretrained_model(backbone_name: str = "efficientnet"):
    backbones = {
        "efficientnet": EfficientNetB0,
        "vgg": VGG16,
        "resnet": ResNet50
    }

    if backbone_name not in backbones:
        raise ValueError(f"backbone_name must be one of {list(backbones.keys())}")

    BackboneClass = backbones[backbone_name]

    base_model = BackboneClass(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )

    model = add_last_layers(base_model)

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )
    return model

# --------------------- SAVE & LOAD -----------------------
def save_model(model, filepath: str):
    model.save(filepath)
    print(f"✅ Model saved to {filepath}")

def load_model(filepath: str):
    model = keras.models.load_model(filepath)
    print(f"✅ Model loaded from {filepath}")
    return model
