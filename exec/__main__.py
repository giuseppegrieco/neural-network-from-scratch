import libraries as ml

hyper_parameters = ml.Hyperparameters(
     2, #input layer
     [
          [3,  ml.sigmoid, ml.sigmoid_derivative]
     ], #hidden layers
     [2, ml.sigmoid, ml.sigmoid_derivative],  #output layer
     0.01
)

nn = ml.NeuralNetwork(hyper_parameters)
nn.train([2, 5], [12, 53])