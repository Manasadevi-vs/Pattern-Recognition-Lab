import tensorflow as tf
from tensorflow.keras import datasets, models
from tensorflow.keras.layers import *

(xtr,ytr),(xte,yte)=datasets.mnist.load_data()
xtr,xte=xtr.reshape(-1,28,28,1)/255.0, xte.reshape(-1,28,28,1)/255.0
ytr,yte=tf.keras.utils.to_categorical(ytr,10), tf.keras.utils.to_categorical(yte,10)

m=models.Sequential([
    Conv2D(32,3,activation='relu',input_shape=(28,28,1)), MaxPooling2D(),
    Conv2D(64,3,activation='relu'), MaxPooling2D(),
    Flatten(), Dense(64,activation='relu'), Dense(10,activation='softmax')
])

m.compile('adam','categorical_crossentropy',['accuracy'])
m.fit(xtr,ytr,epochs=5,validation_data=(xte,yte),batch_size=64)
print("Accuracy:", m.evaluate(xte,yte)[1])
