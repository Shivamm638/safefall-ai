"""
SafeFall AI - the activity-classification CNN.

Architecture: a two-branch convolutional network.

    Branch A - Pose CNN
        The 33 body landmarks are treated as a 1-D signal of length 33 with 4
        channels (x, y, z, visibility).  1-D convolutions slide along the
        kinematic chain, so each filter learns a *local body part pattern*
        (shoulder-elbow-wrist, hip-knee-ankle) that is reused wherever it
        appears - exactly the weight-sharing argument that makes CNNs the right
        family for this problem, applied to skeleton data instead of pixels.

    Branch B - Clinical features
        25 hand-designed descriptors (trunk angle, knee angle, bounding-box
        aspect ratio, stride width, ...).  These give the network the explicit,
        explainable fall cues a physiotherapist would look for and make the
        model's behaviour much easier to defend.

    Head
        The two branches are concatenated and passed through dense layers with
        dropout to a 5-way softmax.

The whole network is ~120 k parameters - small enough to run in real time on a
CPU, which matters because the deployed dashboard has no GPU.
"""

from __future__ import annotations

from typing import Tuple

from . import config


def build_model(
    landmark_shape: Tuple[int, int] = (config.NUM_LANDMARKS, config.LANDMARK_CHANNELS),
    n_geometric: int = config.NUM_GEOMETRIC_FEATURES,
    n_classes: int = config.NUM_CLASSES,
    learning_rate: float = config.LEARNING_RATE,
    seed: int = config.RANDOM_SEED,
    l2: float = config.L2_REGULARISATION,
    dropout_geo: float = 0.30,
    dropout_head1: float = 0.45,
    dropout_head2: float = 0.35,
    width: int = 1,
):
    """Compile and return the two-branch Keras model."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers

    tf.keras.utils.set_random_seed(seed)
    reg = regularizers.l2(l2)
    c1, c2 = int(64 * width), int(128 * width)

    # ---------------- Branch A: convolutions over the skeleton -------------- #
    lm_in = layers.Input(shape=landmark_shape, name="landmarks")
    x = layers.Conv1D(c1, 3, padding="same", activation="relu",
                      kernel_regularizer=reg, name="conv1")(lm_in)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv1D(c2, 3, padding="same", activation="relu",
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(2, name="pool1")(x)
    x = layers.Conv1D(c2, 3, padding="same", activation="relu",
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    avg = layers.GlobalAveragePooling1D(name="gap")(x)
    mx = layers.GlobalMaxPooling1D(name="gmp")(x)
    pose_branch = layers.Concatenate(name="pose_features")([avg, mx])

    # ---------------- Branch B: clinical posture descriptors ---------------- #
    geo_in = layers.Input(shape=(n_geometric,), name="geometry")
    g = layers.Dense(int(64 * width), activation="relu", kernel_regularizer=reg, name="geo_dense")(geo_in)
    g = layers.BatchNormalization(name="geo_bn")(g)
    g = layers.Dropout(dropout_geo, name="geo_drop")(g)

    # ---------------- Fusion head ------------------------------------------- #
    z = layers.Concatenate(name="fusion")([pose_branch, g])
    z = layers.Dense(int(128 * width), activation="relu", kernel_regularizer=reg, name="head1")(z)
    z = layers.Dropout(dropout_head1, name="head_drop1")(z)
    z = layers.Dense(int(64 * width), activation="relu", kernel_regularizer=reg, name="head2")(z)
    z = layers.Dropout(dropout_head2, name="head_drop2")(z)
    out = layers.Dense(n_classes, activation="softmax", name="activity")(z)

    model = models.Model(inputs=[lm_in, geo_in], outputs=out, name="SafeFall_PoseCNN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def summary_text(model) -> str:
    """Model summary as a string, for the report and the dashboard."""
    lines: list[str] = []
    model.summary(print_fn=lines.append, line_length=96)
    return "\n".join(lines)
