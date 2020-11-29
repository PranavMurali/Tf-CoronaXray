import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping

train = 'F:/CovidXray/Tf-CoronaXray/COVID19/Train'
validation ='F:/CovidXray/Tf-CoronaXray/COVID19/Test'
train_neg = os.path.join(train, 'Healthy')
train_pos = os.path.join(train, 'Covid')
validation_neg = os.path.join(validation, 'Healthy')
validation_pos = os.path.join(validation, 'Covid')

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
model.summary() 

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



best_model_file = "Xray-Covid.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', mode='max',verbose=1, save_best_only=True)

earlystop = EarlyStopping(monitor = 'val_loss',min_delta = 0,patience = 3,verbose = 1,restore_best_weights = True)

history = model.fit(train_generator,epochs=15,verbose=1,validation_data=validation_generator,callbacks=[best_model,earlystop])


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

