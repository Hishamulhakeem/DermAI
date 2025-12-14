import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

data_dir_train = './SkinDisease/train'
data_dir_test  = './SkinDisease/test'

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
validation_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    data_dir_train,
    target_size = (224,224),
    batch_size=64,
    class_mode='categorical',
    subset='training'
)

validation_gen = validation_datagen.flow_from_directory(
    data_dir_train,
    target_size = (224,224),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

test_gen = test_datagen.flow_from_directory(
    data_dir_test,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical', 
    shuffle=False
)

model1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(22, activation='softmax')  
    
]) 

model2 = models.Sequential([
    ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg'),
    layers.Dense(512, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(22, activation='softmax')
])

model3 = models.Sequential([
    VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg'),
    layers.Dense(512, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(22, activation='softmax')
])

model1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model3.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model1.fit(
    train_gen,
    epochs=10,
    validation_data=validation_gen,
    steps_per_epoch=train_gen.samples//train_gen.batch_size,
    validation_steps=validation_gen.samples//validation_gen.batch_size
)