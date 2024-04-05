import tensorflow as tf
import pandas as p
import matplotlib.pyplot as plt
import numpy as np
from keras import metrics

#get dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.Sequential()

#flatten gør arrayet til 1 dimensionel
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=9)

predictions = model.predict(x_test)


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
# print(np.argmax(predictions[0]))
#
# plt.imshow(x_test[0], cmap=plt.cm.binary)
# plt.show()

#5 epochs giver en 96% accuracy med 12% loss
#6 98% og 4% eller 96% og 11%
#8 99% og 03% eller 97% og 10%
#9 99% og 2% eller 96% og 12%
#9 24neuron 97% og 6% eller 95% og 14%
#9 SGD 94% 20% og 93% og 22%
#10 epochs giver 99% accuracy med 2% loss, eller 97 og 11 med evaluering

#20 giver 97% og 15% - eller 97% og 14%

#100 99% og 0% eller 96% og 41%

#150+ give 99-100% accuracy og 12-800% loss