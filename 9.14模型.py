def build_model():
    model = tf.keras.Sequential([
        # Lambda layer for decoding Base64 images
        layers.Lambda(
            (
                lambda x: tf.map_fn(
                    process_base64_image,
                    x,
                    fn_output_signature=tf.TensorSpec(shape=(int(img_height), int(img_width), 3), dtype=tf.float32))
            ),
            name='decode_base64_png',
            input_shape=(None,)
        ),

        # CNN Layers
        layers.Conv2D(16, 3, padding='same', activation='gelu', name="conv2d_layer1"),
        layers.MaxPooling2D(name="maxpool_layer1"),

        layers.Conv2D(32, 3, padding='same', activation='gelu', name="conv2d_layer2"),
        layers.MaxPooling2D(name="maxpool_layer2"),

        # Dense Layers
        layers.Flatten(name="flatten_layer"),
        layers.Dense(128, activation='gelu', name="dense_layer1"),

        layers.Dense(1, activation='sigmoid', name="output_layer")
    ])

    return model


model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
