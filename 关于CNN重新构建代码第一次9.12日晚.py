# ... [Your Previous Code for Image Preprocessing, Augmentation, etc.] ...

# Necessary Libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Loading Processed Images

# Set up directories
train_dir = './images/train'
val_dir = './images/val'
test_dir = './images/test'

# Image Parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Data Augmentation and normalization for training
train_image_generator = ImageDataGenerator(rescale=1. / 255)

# Just rescaling for validation and test
val_image_generator = ImageDataGenerator(rescale=1. / 255)
test_image_generator = ImageDataGenerator(rescale=1. / 255)

# Data Loading
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = val_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                       directory=val_dir,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=test_dir,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='binary')


# 2. Model Creation and Compilation

def create_advanced_cnn(input_shape):
    input_layer = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
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


model = create_advanced_cnn((IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Model Training
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=10,  # Adjust as needed
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // BATCH_SIZE
)

# 4. Model Evaluation
test_loss, test_accuracy = model.evaluate(test_data_gen)
print(f'Test accuracy: {test_accuracy}')

# 5. Model Saving
model_path = "./saved_model/my_model"
model.save(model_path)

print("Model saved to", model_path)
