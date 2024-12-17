import numpy as np

# Define the softmax activation function
def softmax(z: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of the input array.

    Args:
        z (np.ndarray): Input array (e.g., logits).

    Returns:
        np.ndarray: Softmax-transformed array, where each element is a probability.
    """
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability.
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)  # Normalize by sum.

# Define the derivative of the softmax function
def softmax_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the softmax function.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Derivative of the softmax function for the input.
    """
    s = softmax(z)  # Compute softmax
    return s * (1 - s)  # Simplified derivative for element-wise case.

# Define the tanh activation function
def tanh(z: np.ndarray) -> np.ndarray:
    """
    Compute the hyperbolic tangent (tanh) activation function.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Tanh-transformed array.
    """
    return np.tanh(z)

# Define the derivative of the tanh function
def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the tanh function.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Derivative of the tanh function for the input.
    """
    return 1 - np.tanh(z)**2

# Define the SimpleRNN class
class SimpleRNN:
    """
    A simple Recurrent Neural Network (RNN) for sequence processing.

    Attributes:
        input_size (int): Size of input vectors.
        hidden_size (int): Size of hidden state vectors.
        output_size (int): Size of output vectors.
        learning_rate (float): Learning rate for training.
        W_xh (np.ndarray): Weights for input-to-hidden connections.
        W_hh (np.ndarray): Weights for hidden-to-hidden connections.
        W_hy (np.ndarray): Weights for hidden-to-output connections.
        b_h (np.ndarray): Biases for hidden state.
        b_y (np.ndarray): Biases for output.
        h_prev (np.ndarray): Hidden state from the previous time step.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, learning_rate: float = 0.01):
        """
        Initialize the SimpleRNN with specified dimensions and hyperparameters.

        Args:
            input_size (int): Dimensionality of input vectors.
            hidden_size (int): Dimensionality of hidden state.
            output_size (int, optional): Dimensionality of output. Defaults to 1.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with small random values
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01

        # Initialize biases as zeros
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

        # Initialize the hidden state to zeros
        self.h_prev = np.zeros((hidden_size, 1))

    def forward(self, x_seq: np.ndarray) -> tuple:
        """
        Perform forward propagation through the RNN for a sequence of inputs.

        Args:
            x_seq (np.ndarray): Input sequence of shape (T, input_size),
                                where T is the number of time steps.

        Returns:
            tuple: 
                - h_seq (np.ndarray): Sequence of hidden states of shape (T, hidden_size, 1).
                - y_seq (np.ndarray): Sequence of outputs of shape (T, output_size, 1).
        """
        h_seq = []  # List to store hidden states at each time step.
        y_seq = []  # List to store outputs at each time step.

        for t in range(x_seq.shape[0]):  # Loop through each time step.
            x_t = x_seq[t].reshape(-1, 1)  # Convert input at time t to a column vector.
            z_t = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, self.h_prev) + self.b_h  # Compute input to activation.
            self.h_prev = tanh(z_t)  # Update hidden state using tanh activation.
            o_t = np.dot(self.W_hy, self.h_prev) + self.b_y  # Compute output at time t.
            y_t = softmax(o_t)  # Apply softmax activation to output.

            # Store results
            h_seq.append(self.h_prev)
            y_seq.append(y_t)

        return np.array(h_seq), np.array(y_seq)

    def backward(self, x_seq: np.ndarray, y_seq: np.ndarray, y_pred_seq: np.ndarray) -> tuple:
        """
        Perform backward propagation through the RNN to compute gradients.

        Args:
            x_seq (np.ndarray): Input sequence of shape (T, input_size).
            y_seq (np.ndarray): True output sequence of shape (T, output_size).
            y_pred_seq (np.ndarray): Predicted output sequence of shape (T, output_size).

        Returns:
            tuple:
                - dW_xh (np.ndarray): Gradient of W_xh.
                - dW_hh (np.ndarray): Gradient of W_hh.
                - dW_hy (np.ndarray): Gradient of W_hy.
                - db_h (np.ndarray): Gradient of b_h.
                - db_y (np.ndarray): Gradient of b_y.
        """
        # Initialize gradients with zeros
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))  # Gradient of hidden state from the next time step.

        # Loop backward through time steps
        for t in reversed(range(x_seq.shape[0])):
            x_t = x_seq[t].reshape(-1, 1)  # Input at time step t as a column vector.
            y_true = y_seq[t].reshape(-1, 1)  # True output at time t.
            y_pred = y_pred_seq[t].reshape(-1, 1)  # Predicted output at time t.

            # Compute gradient of the loss with respect to output
            dy_t = y_pred - y_true

            # Gradients for output layer
            dW_hy += np.dot(dy_t, self.h_prev.T)
            db_y += dy_t

            # Gradient with respect to hidden state
            dh_t = np.dot(self.W_hy.T, dy_t) + dh_next

            # Backpropagate through activation function
            dz_t = dh_t * tanh_derivative(self.h_prev)

            # Gradients for weights and biases
            dW_xh += np.dot(dz_t, x_t.T)
            dW_hh += np.dot(dz_t, self.h_prev.T)
            db_h += dz_t

            # Update gradient for next time step
            dh_next = np.dot(self.W_hh.T, dz_t)

        return dW_xh, dW_hh, dW_hy, db_h, db_y

    def reset_hidden_state(self):
        """
        Reset the hidden state of the RNN to zeros.
        """
        self.h_prev = np.zeros((self.hidden_size, 1))
