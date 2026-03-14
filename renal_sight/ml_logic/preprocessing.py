# ------------------------ IMPORTS ------------------------
import tensorflow as tf

# --------------------- CONFIGURATION ---------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# ---------------- PREPROCESSING FUNCTIONS ----------------
vgg_preprocess = tf.keras.applications.vgg16.preprocess_input
resnet_preprocess = tf.keras.applications.resnet50.preprocess_input
efficientnet_preprocess = tf.keras.applications.efficientnet.preprocess_input

# --------------------- LOAD DATASETS ---------------------
def load_datasets(augmented_dir: str, original_dir: str):

    ## Train dataset - Augmented data
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        augmented_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["Non-Stone", "Stone"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
    )

    ## Full Original Dataset
    full_original = tf.keras.preprocessing.image_dataset_from_directory(
        original_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["Non-Stone", "Stone"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=RANDOM_SEED
    )

    ## Shuffle before split
    full_original_shuffled = full_original.shuffle(
        buffer_size=len(full_original) * BATCH_SIZE,
        seed=RANDOM_SEED,
        reshuffle_each_iteration=False
    )

    ## Val and Test dataset - From Full Original dataset
    total_batches = len(full_original)
    val_batches = total_batches // 2

    val_dataset = full_original_shuffled.take(val_batches).cache().prefetch(AUTOTUNE)
    test_dataset = full_original_shuffled.skip(val_batches).cache().prefetch(AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


# --------------------- GET DATASETS ----------------------
def prepare(dataset, preprocess_input):
    return (
        dataset
        .map(lambda img, label: (preprocess_input(img), label), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

def get_datasets(train_dataset, val_dataset, test_dataset, preprocess_input):
    return (
        prepare(train_dataset, preprocess_input),
        prepare(val_dataset, preprocess_input),
        prepare(test_dataset, preprocess_input)
    )

# ── API IMAGE PREPROCESSING ───────────────────────────────────
def preprocess_image_from_bytes(image_bytes: bytes) -> tf.Tensor:
    """
    Takes raw image bytes from an API upload and returns
    a preprocessed tensor ready for model prediction.
    """
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    return img

def preprocess_image_for_backbone(image_bytes: bytes, backbone: str = "efficientnet") -> tf.Tensor:
    """
    Preprocesses image bytes with backbone-specific normalization.
    backbone: 'efficientnet', 'vgg', 'resnet'
    """
    preprocess_fns = {
        "efficientnet" : efficientnet_preprocess,
        "vgg"          : vgg_preprocess,
        "resnet"       : resnet_preprocess,
    }

    if backbone not in preprocess_fns:
        raise ValueError(f"backbone must be one of {list(preprocess_fns.keys())}")

    img = preprocess_image_from_bytes(image_bytes)
    img = preprocess_fns[backbone](img)
    return img
