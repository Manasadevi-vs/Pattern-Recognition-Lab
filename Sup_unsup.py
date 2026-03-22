import os, cv2, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

X,y=[],[]; path="path_to_your_images"; classes=os.listdir(path)
for i,c in enumerate(classes):
    for f in os.listdir(path+"/"+c):
        img=cv2.imread(path+"/"+c+"/"+f)
        if img is not None:
            X.append(cv2.resize(img,(64,64))/255); y.append(i)

X,y=np.array(X),np.array(y)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2)
ytr,yte=np.eye(len(classes))[ytr],np.eye(len(classes))[yte]

m=Sequential([Conv2D(32,3,activation='relu',input_shape=(64,64,3)),MaxPooling2D(),
              Conv2D(64,3,activation='relu'),MaxPooling2D(),
              Flatten(),Dense(128,activation='relu'),Dropout(0.5),
              Dense(len(classes),activation='softmax')])
m.compile('adam','categorical_crossentropy',['accuracy'])
m.fit(Xtr,ytr,epochs=10,validation_data=(Xte,yte))
print(m.evaluate(Xte,yte)[1])

labels=KMeans(n_clusters=len(classes)).fit_predict(X.reshape(len(X),-1))
for i in range(5): plt.imshow(X[i]); plt.title(labels[i]); plt.axis('off'); plt.show()
