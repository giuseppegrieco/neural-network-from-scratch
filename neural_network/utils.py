import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import json
import neural_network as nn


def split_k_fold(input_list, output_list, k):
    size = int(np.rint(len(input_list) / k))
    folds = []
    for i in range(0, k):
        lr_limit = size * i
        up_limit = (size + size * i) if i != k - 1 else len(input_list)
        folds.append([
            input_list[lr_limit:up_limit],
            output_list[lr_limit:up_limit]
        ])
    return folds


def retrieves_fold_k(folds, k):
    tr_in = []
    tr_out = []
    for i in range(0, len(folds)):
        if i != k - 1:
            tr_in = tr_in + folds[i][0]
            tr_out = tr_out + folds[i][1]
    vn_in = folds[k - 1][0]
    vn_out = folds[k - 1][1]
    return tr_in, tr_out, vn_in, vn_out


def split_training_test(input_list, output_list, tr_size):
    tr_size = int(np.rint(len(input_list) * tr_size))
    tr_in = input_list[:tr_size]
    tr_out = output_list[:tr_size]
    ts_in = input_list[tr_size:]
    ts_out = output_list[tr_size:]
    return tr_in, tr_out, ts_in, ts_out


def save_in_csv(input_list, output_list, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(0, len(input_list)):
            writer.writerow([i + 1] + input_list[i] + output_list[i])


def cup_parser(file_name):
    index = 0
    input_list = []
    output_list = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            value = []
            for i in range(1, 21):
                value.append(float(row[i]))
            input_list.append(value)
            output_list.append([float(row[21]), float(row[22])])
            index = index + 1
        csv_file.close()
        return input_list, output_list


def convert_in_numpy(target):
    if isinstance(target, list) and not isinstance(target[0], list):
        target = [1] + target
        target = np.array(target, dtype=np.dtype('d'))
        return target.reshape((len(target), 1))
    else:
        target = np.array(target, dtype=np.dtype('d'))
        n, m = target.shape
        bias = np.ones((1, m), dtype=np.dtype('d'))
        return np.vstack((bias, target))


def save_graph(tr_errors, v_errors, eta, lambda_reg, alpha_momentum, path, file_name, nodes_number):
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


def monk_parser(file_name):
    input_list = []
    output_list = []
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
        return input_list, output_list


def computes_accuracy(targets, expected_output):
    res = []
    for i in np.nditer(targets):
        if i > 0.5:
            res.append(1.0)
        else:
            res.append(0.0)

    all_wrong_output = np.matrix.sum(abs(np.mat(res, dtype=np.dtype('d')) - expected_output))
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
               directory_name, my_nn.get_number_of_nodes())
    save_graph(tr_errors, v_errors, eta, lambda_reg, alpha_momentum,
               "/grid_search/" + directory_name + "/", directory_name, my_nn.get_number_of_nodes())

    np.save("./grid_search/" + directory_name + "/training_error", np.mat(tr_errors))
    np.save("./grid_search/" + directory_name + "/validation_error", np.mat(v_errors))

    np.save("./grid_search/" + directory_name + "/initial_weights", initial_weights)  # TODO: codice eseguibile
    np.save("./grid_search/" + directory_name + "/final_weights", final_weights)  # TODO: codice eseguibile

    create_output_json(eta, lambda_reg, alpha_momentum, epochs, duration_in_sec, my_nn.get_topology(),
                       './grid_search/' + directory_name + '/')


def read_input(data):
    def get_topology_from_json(input_topologies):
        all_topologies = []
        for topology in input_topologies:
            layers = []
            for layer in topology:
                nodes = layer['nodes']
                if layer['activation_function'] == 'Sigmoid':
                    activation_function = nn.Sigmoid()
                else:
                    activation_function = nn.Identity()  # TODO: add all functions
                layers.append(nn.Layer(nodes=nodes, activation_function=activation_function))
            all_topologies.append(layers)

        return all_topologies

    input_size = data['input_size']
    training_file = data['training_file']
    validation_file = data['validation_file']
    epochs = data['epochs']
    a_eta = data["learning_rate"]
    a_lambda_reg = data["lambda_regularization"]
    a_alpha_momentum = data["alpha_momentum"]
    learning_algorithm = data["learning_algorithm"]  # TODO: check
    topologies = data["topology"]
    a_topology = get_topology_from_json(topologies)
    thread_number = data['thread_number']
    folds = data["folds"]

    return input_size, training_file, validation_file, epochs, a_eta, a_lambda_reg, a_alpha_momentum, learning_algorithm, a_topology, thread_number, folds


def early_stopping(error, min_error, counter, epoch):
    res = False
    if error < min_error:
        min_error = error
    elif epoch > 200:  # TODO: metti una variabile o qualocsa
        if counter == 1:
            res = True
        counter = counter - 1
    return error, min_error, counter, epoch, res
