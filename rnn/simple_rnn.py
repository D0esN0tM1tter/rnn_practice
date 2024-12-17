import numpy as np

def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax activation function"""
    exp_z = np.exp(z - np.max(z))  # For numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def softmax_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of the softmax activation function"""
    s = softmax(z)
    return s * (1 - s)

def tanh(z: np.ndarray) -> np.ndarray:
    """Tanh activation function"""
    return np.tanh(z)

def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of tanh activation function"""
    return 1 - np.tanh(z)**2

class SimpleRNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, learning_rate: float = 0.01):
        """
        Initialize the SimpleRNN model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            output_size (int, optional): Number of output features (default is 1 for binary classification).
            learning_rate (float, optional): Learning rate (default is 0.01).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01

        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

        # Initialize hidden state
        self.h_prev = np.zeros((hidden_size, 1))

    def forward(self, x_t: np.ndarray) -> tuple:
        # ensuring that the input is a column vector : 
        x_t = x_t.reshape(-1, 1) if x_t.ndim == 1 else x_t
        
        # 1: Preactivation input z
        self.z_t = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, self.h_prev) + self.b_h

        # 2: Current state
        self.h_t = tanh(self.z_t)

        # 3: Preactivation output o
        self.o_t = np.dot(self.W_hy, self.h_t) + self.b_y

        # 4: Output using softmax
        self.y_t = softmax(self.o_t)

        # Return the current hidden state and output
        return self.h_t, self.y_t

    def backward(self, x_t: np.ndarray, y_true: np.ndarray) -> tuple:
        """
        Perform back-propagation through the RNN.

        Args:
            x_t (np.ndarray): Input at time step t.
            y_true (np.ndarray): True output at time step t (one-hot encoded).

        Returns:
            tuple: Gradients for the weights and biases.
        """
        # 1: Gradient w.r.t the current prediction
        dy_t = self.y_t - y_true

        # 2: Gradient w.r.t W_hy and b_y
        delta_a_t = dy_t
        dW_hy = np.dot(delta_a_t, self.h_t.T)
        db_y = delta_a_t

        # 3: Gradient w.r.t W_xh and b_h
        dh_t = np.dot(self.W_hy.T, delta_a_t)
        dz_t = dh_t * tanh_derivative(self.z_t)
        dW_xh = np.dot(dz_t, x_t.T)
        db_h = dz_t

        # 4: Gradient w.r.t W_hh
        dW_hh = np.dot(dz_t, self.h_prev.T)

        # Return gradients for all parameters
        return dW_xh, dW_hh, dW_hy, db_h, db_y

    def update_parameters(self, dW_xh: np.ndarray, dW_hh: np.ndarray, dW_hy: np.ndarray, db_h: np.ndarray, db_y: np.ndarray) -> None:
        """
        Update the model parameters using the computed gradients.

        Args:
            dW_xh (np.ndarray): Gradient of the loss w.r.t W_xh.
            dW_hh (np.ndarray): Gradient of the loss w.r.t W_hh.
            dW_hy (np.ndarray): Gradient of the loss w.r.t W_hy.
            db_h (np.ndarray): Gradient of the loss w.r.t b_h.
            db_y (np.ndarray): Gradient of the loss w.r.t b_y.
        """
        self.W_xh -= self.learning_rate * dW_xh
        self.W_hh -= self.learning_rate * dW_hh
        self.W_hy -= self.learning_rate * dW_hy
        self.b_h -= self.learning_rate * db_h
        self.b_y -= self.learning_rate * db_y

    def reset_hidden_state(self) -> None:
        """Reset the hidden state."""
        self.h_prev = np.zeros((self.hidden_size, 1))
