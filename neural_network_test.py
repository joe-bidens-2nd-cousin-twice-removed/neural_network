import numpy as np
from neural_network import NeuralNetwork, ActivationFunction as Activation
import matplotlib.pyplot as plt

rng = np.random.default_rng()

nn = NeuralNetwork()

def generate_xnor_data(num_inputs=4):
    data = []
    inputs = []
    outputs = []

    for i in range(2**num_inputs):
        output = 0
        bits = []

        string = format(i, f"0{num_inputs}b")

        bits = [int(ch) for ch in string]

        if bits.count(1) % 2 == 0:
            output = 1

        inputs.append(bits)
        outputs.append([output])

    data.append(np.array(inputs))
    data.append(np.array(outputs))

    return data

X, Y = generate_xnor_data(5)

test = np.array([0, 1, 1, 1, 0])

nn.add_layer(32, input_dim=5, activation_function=Activation.RELU)
nn.add_layer(32, activation_function=Activation.RELU)
nn.add_layer(1, activation_function=Activation.SIGMOID)

plt.plot(nn.train(X, Y, training_epochs=20000, lr=0.1))
print(nn)
plt.show()