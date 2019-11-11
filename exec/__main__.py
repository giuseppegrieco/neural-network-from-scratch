import libraries as ml
import matplotlib.pyplot as plt

hyper_parameters = ml.Hyperparameters(
    2,  # input layer
    [
        [3, True, ml.sigmoid, ml.sigmoid_derivative],
    ],  # hidden layers
    [1, False, ml.sigmoid, ml.sigmoid_derivative],  # output layer
    0.1
)

nn = ml.NeuralNetwork(hyper_parameters)

# wrong way !!!!

for i in range(1, 2000):
    nn.train([0, 0], [0])
    nn.train([1, 1], [1])
    nn.train([1, 0], [0])
    nn.train([0, 1], [0])

print(nn.feed_forward([0, 0]))
print(nn.feed_forward([1, 0]))
print(nn.feed_forward([0, 1]))
print(nn.feed_forward([1, 1]))

plt.plot(nn.get_errors())
plt.show()
