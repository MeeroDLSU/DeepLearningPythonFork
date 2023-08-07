import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import csv

# Define the activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights and biases
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_dim, self.hidden_dim))
        self.biases_hidden = np.random.uniform(-1, 1, (1, self.hidden_dim))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_dim, self.output_dim))
        self.biases_output = np.random.uniform(-1, 1, (1, self.output_dim))

    # Forward pass
    def forward(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.biases_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.biases_output
        output = sigmoid(output_layer_input)
        return output

    # Train using backpropagation
    def train_backpropagation(self, X, y, learning_rate=0.1, epochs=1000):
        epoch_times = []
        for _ in range(epochs):
            start_time = time.time()
            # Forward pass
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.biases_hidden
            hidden_layer_output = sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.biases_output
            output = sigmoid(output_layer_input)

            # Backward pass
            output_error = y - output
            output_delta = output_error * sigmoid_derivative(output)
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
            self.biases_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
            self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
            self.biases_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

            # Record epoch time
            end_time = time.time()
            epoch_times.append(end_time - start_time)
        return epoch_times

    # Train using centered finite differences for gradients
    def train_centered_finite_differences(self, X, y, learning_rate=0.1, epsilon=1e-7, epochs=1000):
        epoch_times = []

        for _ in range(epochs):
            start_time = time.time()

            original_weights_input_hidden = self.weights_input_hidden.copy()
            original_biases_hidden = self.biases_hidden.copy()
            original_weights_hidden_output = self.weights_hidden_output.copy()
            original_biases_output = self.biases_output.copy()

            # Calculate gradients using centered finite differences
            for i in range(self.input_dim):
                for j in range(self.hidden_dim):
                    self.weights_input_hidden[i][j] += epsilon
                    output_plus_epsilon = self.forward(X)
                    loss_plus_epsilon = 0.5 * np.sum((y - output_plus_epsilon) ** 2)
                    self.weights_input_hidden[i][j] -= 2 * epsilon
                    output_minus_epsilon = self.forward(X)
                    loss_minus_epsilon = 0.5 * np.sum((y - output_minus_epsilon) ** 2)
                    gradient = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
                    self.weights_input_hidden[i][j] = original_weights_input_hidden[i][j] - learning_rate * gradient

            for i in range(self.hidden_dim):
                self.biases_hidden[0][i] += epsilon
                output_plus_epsilon = self.forward(X)
                loss_plus_epsilon = 0.5 * np.sum((y - output_plus_epsilon) ** 2)
                self.biases_hidden[0][i] -= 2 * epsilon
                output_minus_epsilon = self.forward(X)
                loss_minus_epsilon = 0.5 * np.sum((y - output_minus_epsilon) ** 2)
                gradient = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
                self.biases_hidden[0][i] = original_biases_hidden[0][i] - learning_rate * gradient

            for i in range(self.hidden_dim):
                for j in range(self.output_dim):
                    self.weights_hidden_output[i][j] += epsilon
                    output_plus_epsilon = self.forward(X)
                    loss_plus_epsilon = 0.5 * np.sum((y - output_plus_epsilon) ** 2)
                    self.weights_hidden_output[i][j] -= 2 * epsilon
                    output_minus_epsilon = self.forward(X)
                    loss_minus_epsilon = 0.5 * np.sum((y - output_minus_epsilon) ** 2)
                    gradient = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
                    self.weights_hidden_output[i][j] = original_weights_hidden_output[i][j] - learning_rate * gradient

            for i in range(self.output_dim):
                self.biases_output[0][i] += epsilon
                output_plus_epsilon = self.forward(X)
                loss_plus_epsilon = 0.5 * np.sum((y - output_plus_epsilon) ** 2)
                self.biases_output[0][i] -= 2 * epsilon
                output_minus_epsilon = self.forward(X)
                loss_minus_epsilon = 0.5 * np.sum((y - output_minus_epsilon) ** 2)
                gradient = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
                self.biases_output[0][i] = original_biases_output[0][i] - learning_rate * gradient

            # Record epoch time
            end_time = time.time()
            epoch_times.append(end_time - start_time)
        return epoch_times
    
# Plot epoch times
def plot_epoch_times(epoch_times_backpropagation, epoch_times_centered_fd):
    accumulated_times_backpropagation = np.cumsum(epoch_times_backpropagation)
    accumulated_times_centered_fd = np.cumsum(epoch_times_centered_fd)

    df = pd.DataFrame({
        'Epoch': range(1, len(accumulated_times_backpropagation) + 1),
        'Backpropagation': accumulated_times_backpropagation,
        'Centered Finite Differences': accumulated_times_centered_fd
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Backpropagation'], mode='lines', name='Backpropagation'))
    fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Centered Finite Differences'], mode='lines', name='Centered Finite Differences'))

    fig.update_layout(title='Accumulated Epoch Times', xaxis_title='Epoch', yaxis_title='Time (seconds)')
    fig.show()

def lists_to_csv(backpropagation_times, finite_difference_times, filename='output.csv'):
    # Ensure the lists are of the same length
    assert len(backpropagation_times) == len(finite_difference_times), "Lists must be of the same length"
    
    # Prepare data
    data = [['Epoch', 'Backpropagation', 'Centered Finite Difference']]
    for i, (bp_time, fd_time) in enumerate(zip(backpropagation_times, finite_difference_times)):
        data.append([i+1, bp_time, fd_time])
    
    # Write to CSV
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"Data written to {filename}")

# Test the neural networks
def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate some dummy data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create and train the neural network with backpropagation
    print("Training neural network with backpropagation...")
    nn_backpropagation = NeuralNetwork(input_dim=2, hidden_dim=2, output_dim=1)
    epoch_times_backpropagation = nn_backpropagation.train_backpropagation(X, y, learning_rate=0.1, epochs=50000)

    # Create and train the neural network with centered finite differences
    print("\nTraining neural network with centered finite differences...")
    nn_centered_fd = NeuralNetwork(input_dim=2, hidden_dim=2, output_dim=1)
    epoch_times_centered_fd = nn_centered_fd.train_centered_finite_differences(X, y, learning_rate=0.1, epochs=50000)


    print("\nGround Truth:")
    print(y)

    # Print predictions after training
    print("\nPredictions after training:")
    print("Backpropagation predictions: \n", nn_backpropagation.forward(X))
    print("Centered finite differences predictions: \n", nn_centered_fd.forward(X))


    # Plot epoch times and predictions
    plot_epoch_times(epoch_times_backpropagation, epoch_times_centered_fd)


    lists_to_csv(epoch_times_backpropagation, epoch_times_centered_fd)




if __name__ == "__main__":
    main()
