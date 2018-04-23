# Code to set up a simple Keras example -- Code by Kalen

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from random import sample

np.random.seed(7)
test_size = 150
matches = 0

dataset = np.loadtxt('pima.csv', delimiter=',')

# Splitting the data into input and output variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
x_test = []
y_test = []

items = sample(range(0,len(X)), test_size)
items = sorted(items, reverse=True)
for n in items:
    x_test.append(X[n])
    y_test.append(Y[n])
    X = np.delete(X, n, 0)
    Y = np.delete(Y, n, 0)

# Creating the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model and run the fitness function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10, verbose=2)

#Creating predictions based on the network

x_test = np.array(x_test)
predictions = model.predict(x_test)
rounded = [round(x[0]) for x in predictions]

print(rounded)

for n in range(len(rounded)):
    if rounded[n] == y_test[n]:
        matches = matches + 1

print(matches/test_size)
