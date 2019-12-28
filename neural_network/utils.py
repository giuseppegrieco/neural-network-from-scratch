import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import json


def convert_in_numpy(target):
    if isinstance(target, list) and not isinstance(target[0], list):
        target = [1] + target
        target = np.array(target, dtype=float)
        return target.reshape((len(target), 1))
    else:
        target = np.array(target)
        n, m = target.shape
        bias = np.ones((1, m))
        return np.vstack((bias, target))


def save_graph(tr_errors, v_errors, eta, lambda_reg, alpha_momentum, path, file_name,nodes_number):
    fig, subplot = plt.subplots(nrows=1, ncols=1)
    subplot.plot(tr_errors, '-b', label='Training')
    subplot.plot(v_errors, '--r', label='Validation')
    subplot.legend()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text_str = '\n'.join((
        r'hidden layers=' + str(nodes_number),  # TODO: metti il numero di layer giusti
        r'eta=%.2f' % (eta,),
        r'lambda=%.2f' % (lambda_reg,),
        r'alpha=%.2f' % (alpha_momentum,)))
    subplot.text(0.695, 0.8, text_str, transform=subplot.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

    # Random filename
    # all_char = string.ascii_letters + string.digits
    # graph_name = "".join(random.choice(all_char) for x in range(random.randint(8, 12)))

    my_path = os.path.dirname(__file__)
    try:
        fig.savefig(os.path.dirname(my_path) + path + file_name)
    except Exception as e:
        print(e)


def monk_parser(file_name, input_list, output_list):
    index = 0
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')

        for row in csv_reader:
            value = [0] * 17
            value[int(row[2]) - 1] = 1
            value[3 + int(row[3]) - 1] = 1
            value[6 + int(row[4]) - 1] = 1
            value[8 + int(row[5]) - 1] = 1
            value[11 + int(row[6]) - 1] = 1
            value[15 + int(row[7]) - 1] = 1
            input_list.append(value)
            output_list.append(float(row[1]))
            index = index + 1
        csv_file.close()
        return output_list


def computes_accuracy(targets, expected_output):
    res = []
    for i in np.nditer(targets):
        if i > 0.5:
            res.append(1.0)
        else:
            res.append(0.0)

    all_wrong_output = np.matrix.sum(abs(np.mat(res) - expected_output))
    m, n = targets.shape
    return 1.0 - (all_wrong_output / n)


def create_output_json(eta, lambda_reg, alpha_momentum, epochs, duration_in_sec, topology, path):
    data = \
        {
            'learning_algorithm':
                {  # TODO: fix it when the learning algorithm will be a parameter
                    'name': 'gradient_descent',
                    'learning_rate': eta,
                    'lambda_regularization': lambda_reg,
                    'alpha_momentum': alpha_momentum
                },
            'topology': {},
            'epochs': epochs,
            'duration_sec': duration_in_sec
        }

    index = 0
    for layer in topology:
        activation_function_name = layer.get_activation_function()
        activation_function_name = activation_function_name.__class__.__name__
        data['topology'][str(index)] = {'nodes': layer.get_nodes(), 'activation_function': activation_function_name}
        index = index + 1

    with open(path + '/data.json', 'w') as fp:
        json.dump(data, fp)


def save_data(directory_name, tr_errors, v_errors, final_weights, initial_weights, eta, lambda_reg, alpha_momentum,
              epochs, duration_in_sec, my_nn):
    save_graph(tr_errors, v_errors, eta, lambda_reg, alpha_momentum, "/charts/",
               directory_name,my_nn.get_number_of_nodes())
    save_graph(tr_errors, v_errors, eta, lambda_reg, alpha_momentum,
               "/grid_search/" + directory_name + "/", directory_name, my_nn.get_number_of_nodes())

    np.save("./grid_search/" + directory_name + "/training_error", np.mat(tr_errors))
    np.save("./grid_search/" + directory_name + "/validation_error", np.mat(v_errors))

    np.save("./grid_search/" + directory_name + "/initial_weights", initial_weights)  # TODO: codice eseguibile
    np.save("./grid_search/" + directory_name + "/final_weights", final_weights)  # TODO: codice eseguibile

    create_output_json(eta, lambda_reg, alpha_momentum, epochs, duration_in_sec, my_nn.get_topology(),
                       './grid_search/' + directory_name + '/')



