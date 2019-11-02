import libraries as ml
import random

hyper_parameters = ml.Hyperparameters(
     2, #input layer
     [
          [4,  ml.sigmoid, ml.sigmoid_derivative],
          [3,  ml.sigmoid, ml.sigmoid_derivative]
     ], #hidden layers
     [2, ml.sigmoid, ml.sigmoid_derivative],  #output layer
     0.1
)

nn = ml.NeuralNetwork(hyper_parameters)

# wrong way !!!!
x = [[0,0],[1,0],[0,1],[1,1],[1,1],[1,1]]
y = [[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]

for i in range(1, 100000):
     rand = random.randrange(5)
     nn.train(x[rand], y[rand])


print(nn.feed_forward([0, 0]))
print(nn.feed_forward([1, 0]))
print(nn.feed_forward([0, 1]))
print(nn.feed_forward([1, 1]))