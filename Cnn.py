from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

training = ImageDataGenerator(rescale=1 / 255)
validation = ImageDataGenerator(rescale=1 / 255)

training_dataset = training.flow_from_directory('data/training',
                                                target_size=(200, 200),
                                                batch_size=5,
                                                class_mode='binary')
validation_dataset = validation.flow_from_directory('data/validation',
                                                    target_size=(200, 200),
                                                    batch_size=5,
                                                    class_mode='binary')
# print(training_dataset.classes)
cnn = models.Sequential([layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 3)),
                         layers.MaxPooling2D(2, 2),
                         layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
                         layers.MaxPooling2D((2, 2)),
                         layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
                         layers.MaxPooling2D((2, 2)),

                         layers.Flatten(),

                         layers.Dense(512, activation='relu'),
                         layers.Dense(10, activation='softmax')
                         ])

cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn.fit(training_dataset, steps_per_epoch=10, epochs=30, validation_data=validation_dataset)

# print(validation_dataset.class_indices)
datadir = "data/testing"
categories = ["cars", "motorcycles"]

for category in categories:
    path = os.path.join(datadir, category)
    for i in os.listdir(path):
        img = image.load_img(path + '/' + i, target_size=(200, 200))
        # plt.imshow(img)
        # plt.show()
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        val = cnn.predict(images)
        val = np.argmax(val, axis=1)
        # print(val)
        if val == 0:
            plt.imshow(img)
            plt.title('This is a CAR')
            plt.show()
        else:
            plt.imshow(img)
            plt.title('This is a MOTORCYCLE')
            plt.show()




