import libraries as ml

hyper_parameters = ml.Hyperparameters(
     2, #input layer
     [
          [3,  ml.sigmoid, ml.sigmoid_derivative]
     ], #hidden layers
     [1, ml.sigmoid, ml.sigmoid_derivative],  #output layer
     0.01
)

nn = ml.NeuralNetwork(hyper_parameters)

# wrong way !!!!
for i in range(1, 100):
     nn.train([0, 0], [0])

print(nn.feed_forward([0, 0]))