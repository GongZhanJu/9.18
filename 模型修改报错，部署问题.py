import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import os

# Print TensorFlow version
print(tf.__version__)

# Data directories
train_dir = './images/train'
val_dir = './images/val'
test_dir = './images/test'

# Image dimensions
img_height, img_width = 218, 175
input_shape = (img_height, img_width, 3)
BATCH_SIZE = 32

# Data generators from the first program
train_image_generator = ImageDataGenerator(rescale=1. / 255)
val_image_generator = ImageDataGenerator(rescale=1. / 255)
test_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(img_height, img_width),
                                                           class_mode='binary')

val_data_gen = val_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                       directory=val_dir,
                                                       target_size=(img_height, img_width),
                                                       class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=test_dir,
                                                         target_size=(img_height, img_width),
                                                         class_mode='binary')

# Base64 image processing from the second program
def process_base64_image(s):
    img = tf.io.decode_base64(s)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, (img_height, img_width), antialias=True)
    return img

# Advanced CNN model from the first program adapted for base64 images
def create_advanced_cnn_base64():
    input_layer = Input(shape=(), dtype=tf.string)
    x = layers.Lambda(lambda x: tf.map_fn(process_base64_image, x, fn_output_signature=tf.TensorSpec(shape=input_shape, dtype=tf.float32)), name='decode_base64_png')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

model = create_advanced_cnn_base64()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
epochs = 100
history = model.fit(train_data_gen, epochs=epochs, validation_data=val_data_gen)
