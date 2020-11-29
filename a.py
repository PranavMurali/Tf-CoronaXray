import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = './xray_dataset_covid19/train'
validation ='./xray_dataset_covid19/test'
train_neg = os.path.join(train, 'NORMAL')
train_pos = os.path.join(train, 'COVID')
validation_neg = os.path.join(validation, 'NORMAL')
validation_pos = os.path.join(validation, 'COVID')

img_input = layers.Input(shape=(150, 150, 3))
model = tf.keras.models.Sequential([
    layers.Conv2D(16, 3, activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2), 
    layers.Conv2D(64, 3, activation='relu'), 
    layers.MaxPooling2D(2),
    layers.Flatten(), 
    layers.Dense(512, activation='relu'), 
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
        validation,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')

history = model.fit_generator(train_generator,epochs=5,verbose=1,validation_data=validation_generator)
