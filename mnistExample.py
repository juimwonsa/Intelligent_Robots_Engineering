# -*- coding: utf-8 -*-
#import tensorflow as tf  # 해당 모쥴 전체를 가져온다...
#tf.__version__
from tensorflow import keras # 해당 모쥴내에 있는 특정 메소드나 모쥴내 정의된 변수를 가져온다.
                             # 가져온 매소드나 변수명 앞에 모쥴명을 붙이지 않아도 된다. 
#keras.__version__

mnist = keras.datasets.mnist
(X_train0, y_train0), (X_test0, y_test0) = mnist.load_data()

import matplotlib.pylab as plt

plt.figure(figsize=(6, 1))
for i in range(36):
    plt.subplot(3, 12, i+1)
    plt.imshow(X_train0[i], cmap="gray")
    plt.axis("off")
plt.show()

X_train = X_train0.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test0.reshape(10000, 784).astype('float32') / 255.0
print(X_train.shape, X_train.dtype)

y_train0[:5]
from tensorflow.keras.utils import to_categorical  #one-hot encoding....

Y_train = to_categorical(y_train0, 10)
Y_test = to_categorical(y_test0, 10)
Y_train[:5]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(Dense(15, input_dim=784, activation="sigmoid"))
model.add(Dense(20, activation="relu"))
model.add(Dense(30, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))

model.compile(optimizer=SGD(lr=0.9), loss='mean_squared_error', metrics=["accuracy"])

hist = model.fit(X_train, Y_train,
                 epochs=10, batch_size=100,
                 validation_data=(X_test, Y_test),
                 verbose=2)
print(model.metrics_names)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'])
plt.title("loss")
plt.subplot(1, 2, 2)
plt.title("accuracy")
plt.plot(hist.history['acc'], 'b-', label="training")
plt.plot(hist.history['val_acc'], 'r:', label="validation")
plt.legend()
plt.tight_layout()
plt.show()

number = int(input('몇 번째 학습데이터를 학습완료된 신경망에 넣겠습니까?' ))

model.predict(X_test[number:number+1, :])
plt.figure(figsize=(1, 1))
plt.imshow(X_test0[number], cmap=plt.cm.bone_r)
plt.grid(False)
plt.axis("off")
plt.show()

model.save('my_model.hdf5')
del model
from tensorflow.keras.models import load_model

model2 = load_model('my_model.hdf5')

predictnum = model2.predict_classes(X_test[number:number+1, :])
print(predictnum)

import pyttsx3

engine = pyttsx3.init()

engine.setProperty('rate', 150)
rate = engine.getProperty('rate')

engine.setProperty('volume', 1) # 0~1 
volume = engine.getProperty('volume')

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) # 남성

engine.say("당신이 선택한 숫자는 "+str(predictnum)+"입니다") 
engine.runAndWait() 
engine.stop() 



 