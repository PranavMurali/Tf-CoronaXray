import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

train = './datamini/train'
validation ='./datamini/test'
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

history = model.fit(train_generator,epochs=15,verbose=1,validation_data=validation_generator)
print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


def load_image(filename):
        img = load_img(filename, target_size=(150,150))
        img = img_to_array(img)
        img = img.reshape(1,150,150, 3)
        img = img.astype('float32')
        img = img -[123.68, 116.779, 103.939]
        return img

img = load_image('Insert your image path here')
result = model.predict(img)
if(result[0]==1):
        print("COVID-19 positive")
else:
        print("COVID-19 negative")