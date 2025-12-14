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

history2 = model2.fit(
    train_gen,
    epochs=10,
    validation_data=validation_gen,
    steps_per_epoch=train_gen.samples//train_gen.batch_size,
    validation_steps=validation_gen.samples//validation_gen.batch_size
)

model2.save('model.h5')

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)   
img_path = "test/image4.jpeg"

class_labels = [
    "Acne",
    "Actinic_Keratosis",
    "Benign_tumors",
    "Bullous",
    "Candidiasis",
    "DrugEruption",
    "Eczema",
    "Infestations_Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrh_Keratoses",
    "SkinCancer",
    "Sun_Sunlight_Damage",
    "Tinea",
    "Unknown_Normal",
    "Vascular_Tumors",
    "Vasculitis",
    "Vitiligo",
    "Warts"
]

img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model2.predict(img_array, verbose=0)
predicted_index = np.argmax(prediction)
predicted_class = class_labels[predicted_index]
confidence = prediction[0][predicted_index] * 100

plt.imshow(img)
plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
plt.axis("off")
plt.show()

train_acc = history2.history['accuracy'][-1]
val_acc = history2.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {train_acc*100:.2f}%")
print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
