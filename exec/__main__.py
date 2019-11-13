import libraries as ml
import matplotlib.pyplot as plt
import csv
import numpy as np

hyper_parameters = ml.Hyperparameters(
    6,  # input layer
    [
        [5, True, ml.sigmoid, ml.sigmoid_derivative],
    ],  # hidden layers
    [1, False, ml.sigmoid, ml.sigmoid_derivative],  # output layer
    0.1,
    0.0
)

nn = ml.NeuralNetwork(hyper_parameters)

ts = []
vs = []
i = 0
with open('monks-1.train') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    for row in csv_reader:
        if i <= 79:
            ts.append([])
            ts[i].append([
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7])
            ])
            ts[i].append([float(row[1])])
        else:
            vs.append([])
            vs[i - 80].append([
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7])
            ])
            vs[i - 80].append([float(row[1])])
        i = i + 1

    csv_file.close()
# wrong way !!!!

print(len(ts))
print(len(vs))

terrors = []
verrors = []
for i in range(1, 1000):
    current_verror = 0
    current_terror = 0
    for j in range(0, len(ts)):
        current_terror = current_terror + nn.train(ts[j][0], ts[j][1])
    current_terror = 1 / len(ts) * current_terror
    terrors.append(current_terror)
    for j in range(0, len(vs)):
        current_verror = current_verror + np.power(vs[j][1][0] - nn.feed_forward(vs[j][0]).item(0), 2)
    current_verror = 1 / len(vs) * current_verror
    verrors.append(current_verror)

print(nn.feed_forward(ts[0][0]))
print(nn.feed_forward(ts[1][0]))
print(nn.feed_forward(ts[11][0]))
print(nn.feed_forward(ts[12][0]))

plt.plot(terrors, '-b', label='Training')
plt.plot(verrors, '--r', label='Validation')
plt.legend()
plt.show()
