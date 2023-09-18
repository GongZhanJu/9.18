from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply, Activation


def se_block(tensor, ratio=16):
    """Squeeze and Excitation block."""
    channels = tensor.shape[-1]
    se_shape = (1, 1, channels)

    se = GlobalAveragePooling2D()(tensor)
    se = Reshape(se_shape)(se)
    se = Dense(channels // ratio, activation='relu', use_bias=False)(se)
    se = Dense(channels, activation='sigmoid', use_bias=False)(se)

    return Multiply()([tensor, se])


model = tf.keras.Sequential([
    # Image decoding layer
    layers.Lambda(
        (
            lambda x: tf.map_fn(
                process_base64_image,
                x,
                fn_output_signature=tf.TensorSpec(shape=(int(img_height), int(img_width), 3), dtype=tf.float32))
        ),
        name='decode_base64_png'
    ),

    # Depthwise separable convolution layers
    layers.SeparableConv2D(32, 3, padding='same', activation='swish'),
    layers.MaxPooling2D(),
    layers.SeparableConv2D(64, 3, padding='same', activation='swish'),
    layers.MaxPooling2D(),

    # SE-Block
    layers.Lambda(se_block),

    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='swish', name="dense_layer1"),
    layers.Dense(2, activation='softmax', name="output_layer")
])

# Compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
##########################################################################################################################
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.se_shape = (1, 1, input_shape[-1])
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape = tf.keras.layers.Reshape(self.se_shape)
        self.reduce_dense = tf.keras.layers.Dense(input_shape[-1] // self.ratio, activation='relu', use_bias=False)
        self.expand_dense = tf.keras.layers.Dense(input_shape[-1], activation='sigmoid', use_bias=False)
        super(SEBlock, self).build(input_shape)

    def call(self, x):
        se = self.global_pool(x)
        se = self.reshape(se)
        se = self.reduce_dense(se)
        se = self.expand_dense(se)
        return tf.keras.layers.multiply([x, se])

    def compute_output_shape(self, input_shape):
        return input_shape


# ... Previous layers ...

model = tf.keras.Sequential([
    # Image decoding layer
    layers.Lambda(
        (
            lambda x: tf.map_fn(
                process_base64_image,
                x,
                fn_output_signature=tf.TensorSpec(shape=(int(img_height), int(img_width), 3), dtype=tf.float32))
        ),
        name='decode_base64_png'
    ),
    layers.SeparableConv2D(32, 3, padding='same', activation='swish'),
    layers.MaxPooling2D(),
    layers.SeparableConv2D(64, 3, padding='same', activation='swish'),
    layers.MaxPooling2D(),

    # SE-Block
    SEBlock(),  # <-- Here's the custom layer

    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='swish', name="dense_layer1"),
    layers.Dense(2, activation='softmax', name="output_layer")
])
