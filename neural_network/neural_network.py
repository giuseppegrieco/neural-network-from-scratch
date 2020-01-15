
class NeuralNetwork:
    def __init__(self, input_size, *layers):
        self.__input_size = input_size
        self.output_layer = []
        self.__hidden_layers = []

    def feed_forward(self, input_data):
        output = input_data
        for layer in self.__hidden_layers:
            output = layer.computes(output)
        return self.output_layer.computes(output)

    def add_hidden_layer(self, layer):
        self.__hidden_layers.append(
            layer
        )

    def set_output_layer(self, layer):
        self.output_layer.append(
            layer
        )