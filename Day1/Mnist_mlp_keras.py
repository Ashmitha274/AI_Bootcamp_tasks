import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
#print(x_train.shape)
#print(y_train.shape)
#plt.imshow(x_train[0])
#plt.show()
#print(f"label is :{ y_train[0]}")

#normalize
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#to_categorical
print(f"before: label is : {y_train[0] }")
y_train=to_categorical(y_train)
print(f"after: label is : {y_train[0] }")

y_test=to_categorical(y_test)

#architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28)))  #Flatten is a Keras layer that converts multi-dimensional data (like 2D image pixels) into a 1D vector — so it can be passed into fully connected (Dense) layers.
#defining layer and neuron specification 

model.add(Dense(128,activation='relu')) #128 neurons - hyper parameter; acquire the number on testing(not the last o/p layer)
#last layer
model.add(Dense(10,activation='softmax')) #10 neurons 

#compile 
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #“Categorical” → because we’re classifying into discrete categories-- multiclass

#train
#1 epoch = 60,000 / 128 ≈ 469 updates (batches)
#10 epochs = 469 × 10 = 4,690 updates total
result=model.fit(x_train,y_train,epochs=10,batch_size=64) 

#evaluate
loss,accuracy = model.evaluate(x_test,y_test)
print(f"test_loss:{loss}")
print(f"test_accuracy:{accuracy}")
print(result.history.keys())
print(result.history.values())
print(result.history)

plt.plot(result.history['accuracy'], label='Training Accuracy', color='blue', marker='o')
plt.plot(result.history['val_accuracy'], label='Validation Accuracy', color='orange', marker='o')

plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
