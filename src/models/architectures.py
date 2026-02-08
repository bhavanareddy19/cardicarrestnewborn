"""12 diverse DNN architecture factory functions for the ensemble.

Each function returns a compiled tf.keras.Model accepting 10 tabular features
and outputting 3-class softmax probabilities, unless otherwise noted.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Sequential

from src.config import NUM_FEATURES, NUM_CLASSES, BERT_EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Model 1: ShallowWide -- Simple wide baseline
# ---------------------------------------------------------------------------
def build_shallow_wide(lr: float = 0.001) -> Model:
    model = Sequential([
        layers.Input(shape=(NUM_FEATURES,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="ShallowWide")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 2: DeepNarrow -- 6 layers of width 64 with ELU
# ---------------------------------------------------------------------------
def build_deep_narrow(lr: float = 0.0005) -> Model:
    model = Sequential([
        layers.Input(shape=(NUM_FEATURES,)),
        *[layers.Dense(64, activation="elu",
                        kernel_regularizer=regularizers.l2(0.001))
          for _ in range(6)],
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="DeepNarrow")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 3: PyramidBN -- Decreasing pyramid with BatchNormalization + GELU
# ---------------------------------------------------------------------------
def build_pyramid_bn(lr: float = 0.001) -> Model:
    model = Sequential(name="PyramidBN")
    model.add(layers.Input(shape=(NUM_FEATURES,)))
    for units in [512, 256, 128, 64]:
        model.add(layers.Dense(units))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("gelu"))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 4: DiamondSELU -- Expanding then contracting with SELU
# ---------------------------------------------------------------------------
def build_diamond_selu(lr: float = 0.0008) -> Model:
    model = Sequential(name="DiamondSELU")
    model.add(layers.Input(shape=(NUM_FEATURES,)))
    for units in [64, 128, 256, 128, 64]:
        model.add(layers.Dense(
            units, activation="selu",
            kernel_initializer="lecun_normal",
        ))
        model.add(layers.AlphaDropout(0.1))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 5: ResidualBlock -- Skip connections (Functional API)
# ---------------------------------------------------------------------------
def build_residual_block(lr: float = 0.001) -> Model:
    inputs = layers.Input(shape=(NUM_FEATURES,))
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(0.0005))(inputs)

    # Residual block 1
    shortcut = x
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    # Residual block 2
    shortcut = x
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ResidualBlock")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 6: SwishLayerNorm -- Swish activation with LayerNormalization
# ---------------------------------------------------------------------------
def build_swish_layernorm(lr: float = 0.001) -> Model:
    model = Sequential(name="SwishLayerNorm")
    model.add(layers.Input(shape=(NUM_FEATURES,)))
    for units in [192, 192, 96]:
        model.add(layers.Dense(units))
        model.add(layers.LayerNormalization())
        model.add(layers.Activation("swish"))
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr, weight_decay=0.01
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 7: MixedActivation -- Different activations per layer
# ---------------------------------------------------------------------------
def build_mixed_activation(lr: float = 0.0007) -> Model:
    model = Sequential(name="MixedActivation")
    model.add(layers.Input(shape=(NUM_FEATURES,)))

    activations = ["leaky_relu", "elu", "gelu", "swish", "relu"]
    widths = [128, 96, 64, 48, 32]
    for i, (units, act) in enumerate(zip(widths, activations)):
        model.add(layers.Dense(units, activation=act))
        if i in (1, 3):  # Dropout between layers 2-3 and 4-5
            model.add(layers.Dropout(0.25))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 8: HeavyRegularization -- Strong L1L2 + heavy Dropout
# ---------------------------------------------------------------------------
def build_heavy_regularization(lr: float = 0.0005) -> Model:
    reg = regularizers.l1_l2(l1=0.001, l2=0.001)
    model = Sequential(name="HeavyRegularization")
    model.add(layers.Input(shape=(NUM_FEATURES,)))

    for units, drop in [(256, 0.5), (128, 0.4), (64, 0.3)]:
        model.add(layers.Dense(units, activation="relu",
                               kernel_regularizer=reg))
        model.add(layers.Dropout(drop))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 9: AttentionNet -- TabNet-inspired learned feature attention
# ---------------------------------------------------------------------------
def build_attention_net(lr: float = 0.001) -> Model:
    inputs = layers.Input(shape=(NUM_FEATURES,))

    # Attention branch: learn which features matter
    attn = layers.Dense(64, activation="relu")(inputs)
    attn = layers.Dense(NUM_FEATURES, activation="sigmoid")(attn)

    # Apply attention weights element-wise
    attended = layers.Multiply()([inputs, attn])

    # Processing branch
    x = layers.Dense(128, activation="relu")(attended)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="AttentionNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 10: VeryDeep -- 8 hidden layers with PReLU
# ---------------------------------------------------------------------------
def build_very_deep(lr: float = 0.0003) -> Model:
    model = Sequential(name="VeryDeep")
    model.add(layers.Input(shape=(NUM_FEATURES,)))

    widths = [128, 128, 96, 96, 64, 64, 48, 32]
    for i, units in enumerate(widths):
        model.add(layers.Dense(units))
        model.add(layers.PReLU())
        if i % 2 == 1:  # BatchNorm every 2 layers
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.15))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 11: EmbeddingNet -- Entity embeddings for categorical features
# ---------------------------------------------------------------------------
def build_embedding_net(lr: float = 0.001) -> Model:
    # Each feature is an integer in {1, 2, 3} -> embedding dim 3
    input_layers = []
    embedding_outputs = []

    for i in range(NUM_FEATURES):
        inp = layers.Input(shape=(1,), name=f"feat_{i}")
        input_layers.append(inp)
        # input_dim=4 because values are 1,2,3 (0 unused but keeps indexing simple)
        emb = layers.Embedding(input_dim=4, output_dim=3)(inp)
        emb = layers.Flatten()(emb)
        embedding_outputs.append(emb)

    x = layers.Concatenate()(embedding_outputs)  # 10 * 3 = 30-dim
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=input_layers, outputs=outputs, name="EmbeddingNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Model 12: BERTFusion -- Tabular + BERT embeddings hybrid
# ---------------------------------------------------------------------------
def build_bert_fusion(lr: float = 0.0005) -> Model:
    # Tabular branch
    tabular_input = layers.Input(shape=(NUM_FEATURES,), name="tabular_input")
    tab = layers.Dense(64, activation="relu")(tabular_input)
    tab = layers.Dense(32, activation="relu")(tab)

    # BERT embedding branch
    bert_input = layers.Input(shape=(BERT_EMBEDDING_DIM,), name="bert_input")
    bert = layers.Dense(128, activation="relu")(bert_input)
    bert = layers.Dropout(0.3)(bert)
    bert = layers.Dense(64, activation="relu")(bert)

    # Merge
    merged = layers.Concatenate()([tab, bert])  # 32 + 64 = 96
    x = layers.Dense(64, activation="gelu")(merged)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(
        NUM_CLASSES, activation="softmax",
        kernel_regularizer=regularizers.l2(0.001),
    )(x)

    model = Model(
        inputs=[tabular_input, bert_input], outputs=outputs, name="BERTFusion"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Registry: All 12 model builders
# ---------------------------------------------------------------------------
TABULAR_MODEL_BUILDERS = [
    build_shallow_wide,
    build_deep_narrow,
    build_pyramid_bn,
    build_diamond_selu,
    build_residual_block,
    build_swish_layernorm,
    build_mixed_activation,
    build_heavy_regularization,
    build_attention_net,
    build_very_deep,
]

EMBEDDING_MODEL_BUILDER = build_embedding_net
BERT_FUSION_MODEL_BUILDER = build_bert_fusion

ALL_MODEL_NAMES = [
    "ShallowWide", "DeepNarrow", "PyramidBN", "DiamondSELU",
    "ResidualBlock", "SwishLayerNorm", "MixedActivation",
    "HeavyRegularization", "AttentionNet", "VeryDeep",
    "EmbeddingNet", "BERTFusion",
]
