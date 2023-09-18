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

    # 1st Convolutional Layer
    layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    layers.BatchNormalization(),

    # 2nd Convolutional Layer
    layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding="same"),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    layers.BatchNormalization(),

    # 3rd Convolutional Layer
    layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding="same"),

    # 4th Convolutional Layer
    layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding="same"),

    # 5th Convolutional Layer
    layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    layers.BatchNormalization(),

    # Flattening
    layers.Flatten(),

    # 1st Fully Connected Layer
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),

    # 2nd Fully Connected Layer
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),

    # Output Layer for binary classification
    layers.Dense(1, activation='sigmoid', name="output_layer")
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
