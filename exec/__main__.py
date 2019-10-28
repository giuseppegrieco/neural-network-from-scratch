import libraries as ml

hyperParameters = ml.Hyperparameters(
     5,
     6,
     [2, 3, 5],
     0.01
)

nn = ml.NeuralNetwork(hyperParameters)
