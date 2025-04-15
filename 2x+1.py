import tensorflow as tf #tensorflow library(used for building and training neural networks)
import numpy as np #numpy library(handles numerical data)

X = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float) #inputs (samples)
Y = np.array([1, 3, 5, 7, 9, 11, 13, 15], dtype=float) #expected outputs (2x+1)

model = tf.keras.Sequential([  #initializing sequential model, states layers are sequential(one after another)
    tf.keras.layers.Dense(units=1, input_shape=[1]) #This neural network has one layer(one neuron) and takes one number at a time
])

model.compile(optimizer='sgd', loss ='mean_squared_error') #defines how the model updates its weights and calculates error, aims to minimize error/loss over iterations

model.fit(X, Y, epochs=1000, batch_size = 1, verbose = 0) #trains model by repeatedly learning from number of epochs, verbose controls output, to show the loss after each epoch

new_input = 100 #the input we want to predict
prediction = model.predict(np.array([new_input], dtype = float)) #Defines prediction variable, with function that converts input to array to compute model output
print(f"The answer to {new_input} is: {prediction}") #print our prediction