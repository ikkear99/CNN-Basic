from _ast import mod

from keras.activations import sigmoid
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_applications.nasnet import models

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam' ,loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'] )

model.fit(x_train, y_train, epochs=3)


# plt.imshow(x_train[1], cmap = plt.cm.binary)
# plt.show()
# print(x_train[1])
