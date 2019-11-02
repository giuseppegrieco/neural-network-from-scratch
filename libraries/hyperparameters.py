from layer import _Layer

class Hyperparameters:
    def __init__(self,
                 input_nodes,
                 hidden_layers,
                 output_layer,
                 learning_rate):
        self.__input_nodes = input_nodes
        self.__hidden_layers = []
        self.set_hidden_layers(hidden_layers)
        self.__output_layer = _Layer(
            output_layer[1],
            self.__hidden_layers[-1].get_nodes(),
            output_layer[0],
            output_layer[2],
            output_layer[3]
        )
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

    def set_hidden_layers(self, hidden_layers):
        self.__hidden_layers = []
        previous_nodes = self.__input_nodes
        for layer_info in hidden_layers:
            self.__hidden_layers.append(
                _Layer(
                    layer_info[1],
                    previous_nodes,
                    layer_info[0],
                    layer_info[2],
                    layer_info[3]
                )
            )
            previous_nodes = self.__hidden_layers[-1].get_nodes()

    def set_learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate
