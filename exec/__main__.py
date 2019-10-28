import libraries as ml

hyper_parameters = ml.Hyperparameters(
     5, #input layer
     [[2, ml.sigmoid], [3,  ml.sigmoid], [5,  ml.sigmoid]], #hidden layers
     [6, ml.relu],  #output layer
     0.01
)

nn = ml.NeuralNetwork(hyper_parameters)
nn.computes([1, 2, 3, 4, 5])