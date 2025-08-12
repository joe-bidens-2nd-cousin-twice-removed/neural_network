import numpy as np
from numpy import ndarray
from enum import Enum

_rng = np.random.default_rng(42)
_n = _rng.normal

def _sigmoid(s: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-s))

def _relu(s: ndarray) -> ndarray:
    return np.maximum(s, 0)

def _soft_max(s: ndarray) -> ndarray:
    exp_sum = np.exp(s).sum()

    return np.exp(s) / exp_sum

def _sigmoid(s: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-s))

def _tanh(s: ndarray) -> ndarray:
    return np.tanh(s)

def _sigmoid_deriv(s: ndarray) -> ndarray:  # s is sigmoid(z)
    return s * (1 - s)

def _relu_deriv(s: ndarray) -> ndarray:     # s is relu(z)
    return (s > 0).astype(float)

def _soft_max_deriv(s: ndarray) -> ndarray:
    # S: (N, K), softmax outputs
    N, K = s.shape
    # diag term: for each n, diag(S[n])
    diag_terms = np.einsum('ni,ij->nij', s, np.eye(K))
    # outer term: s s^T per sample
    outer_terms = np.einsum('ni,nj->nij', s, s)
    return diag_terms - outer_terms  # shape (N, K, K)

def _tanh_deriv(s: ndarray) -> ndarray:
    return 1.0 - np.power(s, 2)

def _get_derivative(activation_fn, act):
    if activation_fn == ActivationFunction.SIGMOID:
        return _sigmoid_deriv(act)
    
    elif activation_fn == ActivationFunction.RELU:
        return _relu_deriv(act)
    
    elif activation_fn == ActivationFunction.SOFT_MAX:
        return _soft_max_deriv(act)
    
    elif activation_fn == ActivationFunction.TANH:
        return _tanh_deriv(act)
    
    elif activation_fn == ActivationFunction.NONE:
        return 1
    
    else:
        raise NeuralNetworkException("Unknown activation function")

class NeuralNetworkException(Exception):
    pass

class ActivationFunction(Enum):
    SIGMOID = _sigmoid
    RELU = _relu
    SOFT_MAX = _soft_max
    TANH = _tanh
    NONE = lambda x : x

class NeuralNetwork():
    def __init__(self: "NeuralNetwork"):
        self.layers: list[tuple[int, ActivationFunction]] = []

        self.weights: list[ndarray] = []
        self.biases: list[ndarray] = []

        self.input_neurons: int = 0

    def __repr__(self: "NeuralNetwork"):
        return f"""NeuralNetwork(
            layers={self.layers}
            input_neurons={self.input_neurons}

            weights={self.weights}
            biases={self.biases}
        )"""

    def add_layer(self: "NeuralNetwork", neurons: int=1, input_dim: int=0, activation_function: ActivationFunction = ActivationFunction.SIGMOID) -> None:
        """
        Adds a layer to the neural network

        Parameters
        ----------
        neurons : int
            Number of neurons
        input_dim : int
            Dimensions of input layer
        activation_function : ActivationFunction
            The function to use for each layer
        """
        global _n
        n = _n

        prev_layer: int = 0

        if input_dim:
            self.input_neurons = input_dim
            prev_layer = input_dim
        else:
            prev_layer, _ = self.layers[-1]

        new_weights = n(0, 1, size=(prev_layer, neurons))
        new_biases = n(0, 1, size=neurons)

        self.layers.append((neurons, activation_function))
        self.weights.append(new_weights)
        self.biases.append(new_biases)

    def feed_forward(self: "NeuralNetwork", input: ndarray) -> ndarray:
        """
        Feeds data into the nn

        Parameters
        ----------
        input : ndarray
            The input for the model to predict

        Returns
        -------
        ndarray
            The model's prediction

        Raises
        ------
        NeuralNetworkException
            When there is no provided activation_function
        """

        next: ndarray = input

        for ((_, activation_function), weights, biases) in zip(self.layers, self.weights, self.biases):
            if not callable(activation_function):
                raise NeuralNetworkException()

            z = np.dot(next, weights) + biases
            next = activation_function(z)
            
        return next
    
    # def backpropagate():

    def train(self: "NeuralNetwork", X: ndarray, Y: ndarray, training_epochs: int = 3000, lr: float = 0.1) -> list[float]:
        """
        Trains the neural network using batch gradient descent.

        Parameters
        ----------
        X : ndarray
            Training inputs, shape (batch_size, input_dim)
        Y : ndarray
            Training targets, shape (batch_size, output_dim)
        training_epochs : int
            Number of epochs
        lr : float
            Learning rate

        Returns
        -------
        list[float]
            Loss curve over epochs
        """
        loss_curve: list[float] = []
        batch_size: int = X.shape[0]

        for epoch in range(training_epochs):
            # Forward pass
            activations = [X]
            zs = []
            for (_, activation_function), weights, biases in zip(self.layers, self.weights, self.biases):
                z = np.dot(activations[-1], weights) + biases
                zs.append(z)

                a = activation_function(z)
                activations.append(a)

            # Compute loss
            error = activations[-1] - Y
            loss = np.mean(np.square(error))
            loss_curve.append(loss)

            # Backward pass
            deltas = [error * _get_derivative(self.layers[-1][1], activations[-1])] # Get delta for output layer

            for i in range(len(self.layers) - 2, -1, -1):
                activation_function = self.layers[i][1]
                derivative = _get_derivative(activation_function, activations[i+1])

                delta = (deltas[0] @ self.weights[i+1].T) * derivative
                deltas.insert(0, delta)

            # Update weights and biases
            for i in range(len(self.weights)):
                a_prev = activations[i]
                delta = deltas[i]

                grad_w = a_prev.T @ delta / batch_size
                grad_b = np.mean(delta, axis=0)

                self.weights[i] -= lr * grad_w
                self.biases[i] -= lr * grad_b

            if epoch % max(1, training_epochs // 10) == 0:
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

        return loss_curve

    
    # def train(self: "NeuralNetwork", X: ndarray, Y: ndarray, training_epochs: int = 3000, lr: float = 0.1) -> list[float]:
    #     """
    #     Trains the neural network using batch gradient descent.

    #     Parameters
    #     ----------
    #     X : ndarray
    #         Training inputs, shape (batch_size, input_dim)
    #     Y : ndarray
    #         Training targets, shape (batch_size, output_dim)
    #     training_epochs : int
    #         Number of epochs
    #     lr : float
    #         Learning rate

    #     Returns
    #     -------
    #     list[float]
    #         Loss curve over epochs
    #     """

    #     loss_curve: list[float] = []
    #     batch_size: int = X.shape[0]

    #     for epoch in range(training_epochs):
    #         # Forward pass
    #         activations = [X]
    #         zs = []
    #         for (neurons, activation_function), weights, biases in zip(self.layers, self.weights, self.biases):
    #             z = np.dot(activations[-1], weights) + biases
    #             zs.append(z)
    #             a = activation_function(z)
    #             activations.append(a)

    #         # Compute loss
    #         error = activations[-1] - Y
    #         loss = np.mean(np.square(error))
    #         loss_curve.append(loss)

    #         # Backward pass
    #         deltas = [error * _get_derivative(self.layers[-1][1], activations[-1])]
    #         for i in range(len(self.layers) - 2, -1, -1):
    #             activation_function = self.layers[i][1]
    #             derivative = _get_derivative(activation_function, activations[i+1])
    #             delta = (deltas[0] @ self.weights[i+1].T) * derivative
    #             deltas.insert(0, delta)

    #         # Update weights and biases
    #         for i in range(len(self.weights)):
    #             a_prev = activations[i]
    #             delta = deltas[i]
    #             grad_w = a_prev.T @ delta / batch_size
    #             grad_b = np.mean(delta, axis=0)
    #             self.weights[i] -= lr * grad_w
    #             self.biases[i] -= lr * grad_b

    #         if epoch % max(1, training_epochs // 10) == 0:
    #             print(f"Epoch {epoch}, Loss: {loss:.5f}")

    #     return loss_curve

    # def train(self: "NeuralNetwork", X: ndarray, Y: ndarray, training_epochs: int = 3000, lr: float = 0.1) -> None:
    #     """
    #     Function for training the nn

    #     Parameters
    #     ----------
    #     X : ndarray
    #         The inputs used
    #     Y : ndarray
    #         The expected outputs
    #     training_epochs : int
    #         The number of epochs to train the model
    #     lr : float
    #         The learning rate of the model
    #     """

    #     loss_curve: list[float] = []
    #     batch_size: int = X.shape[0]

    #     activations: list[ndarray] = []
    #     z: list[ndarray] = []
    #     next: ndarray = X

    #     # curr_z: ndarray

    #     # error: ndarray
    #     delta: ndarray
    #     deltas: list[ndarray] = []
    #     # last_delta: ndarray

    #     # derivative: ndarray

    #     for epoch in range(0, training_epochs):
    #         for i, ((_, activation_function), weights, biases) in enumerate(zip(self.layers, self.weights, self.biases)):
    #             if not callable(activation_function):
    #                 raise NeuralNetworkException()

    #             curr_z = np.dot(next, weights) + biases
    #             z.append(curr_z)

    #             next = activation_function(curr_z)
    #             activations.append(next)

    #         error = activations[-1] - Y
    #         loss = np.mean(np.square(error))
    #         loss_curve.append(loss)


    #         for i in range(len(self.layers) - 1, -1, -1):
    #             curr_z = z[i]
    #             activation_function = self.layers[i][1]
    #             derivative = _get_derivative(activation_function, activations[i])

    #             if i == len(self.layers) - 1:
    #                 delta = error * derivative
    #             else:
    #                 last_weights = self.weights[i + 1]
    #                 delta = (last_delta @ last_weights.T) * derivative

    #             deltas.insert(0, delta)

    #             #print(delta)

    #             last_delta = delta

    #         for i in range(0, len(self.weights)):
    #             grad_w = activations[i].T @ deltas[i] / batch_size   # shape: (prev_layer, neurons)
    #             grad_b = np.mean(deltas[i], axis=0)                  # shape: (neurons,)
    #             print(self.weights[i], grad_w, grad_b, deltas[i], activations[i])
    #             print(len(deltas[i]), len(activations))

    #             self.weights[i] -= lr * grad_w
    #             self.biases[i] -= lr * grad_b

    #         # Optionally print loss
    #         if epoch % max(1, training_epochs // 10) == 0:
    #             print(f"Epoch {epoch}, Loss: {loss:.5f}")

    #         activations = []
    #         z = []
    #         next = X
    #         deltas = []

    #     return loss_curve
        
        


