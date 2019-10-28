from layer import Layer


class Hyperparameters:
    def __init__(self,
                 input_nodes,
                 hidden_layers,
                 output_layer,
                 learning_rate):
        self.input_nodes = input_nodes
        self.hidden_layers = []
        self.setHiddenLayers(input_nodes, hidden_layers)
        self.output_layer = Layer(hidden_layers[-1][0], output_layer[0], output_layer[1])
        self.learning_rate = learning_rate

    def get_input_nodes(self):
        return self.input_nodes

    def get_hidden_layers(self):
        return self.hidden_layers

    def get_output_layer(self):
        return self.output_layer

    def get_learning_rate(self):
        return self.learning_rate

    def set_input_nodes(self, input_nodes):
        self.input_nodes = input_nodes

    def set_output_layers(self, output_layers):
        self.output_layer = output_layers

    def setHiddenLayers(self, input_nodes, hidden_layers):
        self.hidden_layers = []
        previous_nodes = input_nodes
        for layer_info in hidden_layers:
            self.hidden_layers.append(
                Layer(previous_nodes, layer_info[0], layer_info[1])
            )
            previous_nodes = layer_info[0]

    def setLearningRate(self, learning_rate):
        self.learning_rate = learning_rate
