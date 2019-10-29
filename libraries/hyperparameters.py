from layer import _Layer

class Hyperparameters:
    """
    Neural Network Hyperparameters abstraction.
    """
    def __init__(self,
                 input_nodes,
                 hidden_layers,
                 output_layer,
                 learning_rate):
        self.__input_nodes = input_nodes
        self.__hidden_layers = []
        self.set_hidden_layers(input_nodes, hidden_layers)
        self.__output_layer = _Layer(hidden_layers[-1][0], output_layer[0], output_layer[1])
        self.__learning_rate = learning_rate

    def get_input_nodes(self):
        return self.__input_nodes

    def get_hidden_layers(self):
        return self.__hidden_layers

    def get_output_layer(self):
        return self.__output_layer

    def get_learning_rate(self):
        return self.__learning_rate

    def set_input_nodes(self, input_nodes):
        self.__input_nodes = input_nodes

    def set_output_layers(self, output_layers):
        self.__output_layer = output_layers

    def set_hidden_layers(self, input_nodes, hidden_layers):
        self.__hidden_layers = []
        previous_nodes = input_nodes
        for layer_info in hidden_layers:
            self.__hidden_layers.append(
                _Layer(previous_nodes, layer_info[0], layer_info[1])
            )
            previous_nodes = layer_info[0]

    def set_learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate
