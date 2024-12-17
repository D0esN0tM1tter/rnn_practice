import numpy as np

# Step 1: Define the softmax activation function
def softmax(z: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of the input array.

    Args:
        z (np.ndarray): Input array (e.g., logits).

    Returns:
        np.ndarray: Softmax-transformed array, where each element is a probability.
    """
    # 1.1 Subtract the maximum value in the array for numerical stability.
    exp_z = np.exp(z - np.max(z))
    # 1.2 Normalize by dividing each element by the sum of the array elements.
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Step 2: Define the derivative of the softmax function
def softmax_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the softmax function.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Derivative of the softmax function for the input.
    """
    # 2.1 Compute the softmax activation for the input.
    s = softmax(z)
    # 2.2 Return the element-wise derivative using the formula s * (1 - s).
    return s * (1 - s)

# Step 3: Define the tanh activation function
def tanh(z: np.ndarray) -> np.ndarray:
    """
    Compute the hyperbolic tangent (tanh) activation function.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Tanh-transformed array.
    """
    # 3.1 Use numpy's tanh function to transform the input.
    return np.tanh(z)

# Step 4: Define the derivative of the tanh function
def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the tanh function.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Derivative of the tanh function for the input.
    """
    # 4.1 Compute the derivative using the formula 1 - tanh(z)^2.
    return 1 - np.tanh(z)**2

# Step 5: Define the SimpleRNN class
class SimpleRNN:
    """
    A simple Recurrent Neural Network (RNN) for sequence processing.
    """

    # Step 5.1: Initialize the RNN with input, hidden, and output sizes
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, learning_rate: float = 0.01):
        """
        Initialize the SimpleRNN with specified dimensions and hyperparameters.

        Args:
            input_size (int): Dimensionality of input vectors.
            hidden_size (int): Dimensionality of hidden state.
            output_size (int, optional): Dimensionality of output. Defaults to 1.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
        """
        # 5.1.1 Set the dimensions and learning rate.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 5.1.2 Initialize weights and biases with small random values and zeros, respectively.
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

        # 5.1.3 Initialize the hidden state as zeros.
        self.h_prev = np.zeros((hidden_size, 1))

    # Step 5.2: Forward propagation
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
        # 5.2.1 Initialize lists to store hidden states and outputs.
        h_seq = []
        y_seq = []

        # 5.2.2 Loop through each time step.
        for t in range(x_seq.shape[0]):
            # 5.2.2.1 Reshape input at time t into a column vector.
            x_t = x_seq[t].reshape(-1, 1)
            # 5.2.2.2 Compute the input to the activation function.
            z_t = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, self.h_prev) + self.b_h
            # 5.2.2.3 Update the hidden state using tanh activation.
            self.h_prev = tanh(z_t)
            # 5.2.2.4 Compute the output using the output layer weights and biases.
            o_t = np.dot(self.W_hy, self.h_prev) + self.b_y
            # 5.2.2.5 Apply the softmax activation to the output.
            y_t = softmax(o_t)

            # 5.2.2.6 Store the hidden state and output for this time step.
            h_seq.append(self.h_prev)
            y_seq.append(y_t)

        # 5.2.3 Convert hidden state and output lists to numpy arrays and return.
        return np.array(h_seq), np.array(y_seq)

    # Step 5.3: Backward propagation
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
        # 5.3.1 Initialize gradients as zero arrays.
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros((self.hidden_size, 1))

        # 5.3.2 Loop backward through time steps.
        for t in reversed(range(x_seq.shape[0])):
            # 5.3.2.1 Reshape input, true output, and predicted output at time t.
            x_t = x_seq[t].reshape(-1, 1)
            y_true = y_seq[t].reshape(-1, 1)
            y_pred = y_pred_seq[t].reshape(-1, 1)

            # 5.3.2.2 Compute the loss gradient with respect to the output.
            dy_t = y_pred - y_true

            # 5.3.2.3 Update gradients for the output layer.
            dW_hy += np.dot(dy_t, self.h_prev.T)
            db_y += dy_t

            # 5.3.2.4 Compute the gradient for the hidden state.
            dh_t = np.dot(self.W_hy.T, dy_t) + dh_next

            # 5.3.2.5 Backpropagate through the tanh activation function.
            dz_t = dh_t * tanh_derivative(self.h_prev)

            # 5.3.2.6 Update gradients for weights and biases.
            dW_xh += np.dot(dz_t, x_t.T)
            dW_hh += np.dot(dz_t, self.h_prev.T)
            db_h += dz_t

            # 5.3.2.7 Update gradient for the next time step.
            dh_next = np.dot(self.W_hh.T, dz_t)

        # 5.3.3 Return computed gradients.
        return dW_xh, dW_hh, dW_hy, db_h, db_y

    # Step 5.4: Reset the hidden state
    def reset_hidden_state(self):
        """
        Reset the hidden state of the RNN to zeros.
        """
        # 5.4.1 Set the hidden state to zeros.
        self.h_prev = np.zeros((self.hidden_size, 1))
