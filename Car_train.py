# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image
from functools import partial
import cv2
import random

inputnum = []
target = []
num_list = list(range(1000))
random.shuffle(num_list)

for i in range(10):
    for j in range(10):
        for k in range(10):
            image = cv2.imread('car'+str(i)+str(j)+str(k)+'.jpg')
            image1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            ret,bin_image = cv2.threshold(image1,127,255,cv2.THRESH_BINARY)
            resimage = cv2.resize(bin_image,(28, 28))/255

            inputnum.append(resimage)
            if i in (0,1):
                des_y = 0
            elif i in(2,3):
                des_y = 1
            elif i in(4,5):
                des_y = 2
            elif i in(6,7):
                des_y = 3
            else:
                des_y = 4
            target.append(des_y)

x = []
y = []
for i in num_list:
    x.append(inputnum[i])
    y.append(target[i])

X_train_full = np.array(x)
y_train_full = np.array(y)

inputnum = []
target = []

for i in range(1000,1250):
    image = cv2.imread('car'+str(i)+'.jpg')
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,bin_image = cv2.threshold(image1,127,255,cv2.THRESH_BINARY)
    resimage = cv2.resize(bin_image,(28, 28))/255

    inputnum.append(resimage)
    if i in range(1000,1050):
        des_y = 1
    elif i in range(1050,1100):
        des_y = 2
    elif i in range(1100,1150):
        des_y = 3
    elif i in range(1150,1200):
        des_y = 4
    else:
        des_y = 0
    target.append(des_y)
X_test = np.array(inputnum)
y_test = np.array(target)



#(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:800], X_train_full[800:]
y_train, y_valid = y_train_full[:800], y_train_full[800:]


X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=5, activation='softmax'),
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#score = model.evaluate(X_test, y_test)
score = model.evaluate(X_test, y_test, verbose=1)
print('Test accuracy:', score[1])



X_new = X_test[:10] # pretend we have new images
y_pred = model.predict(X_new)

y_pred_index = []
for i in range(len(y_pred)):
    y_pred_index.append(np.argmax(y_pred[i]))
    


