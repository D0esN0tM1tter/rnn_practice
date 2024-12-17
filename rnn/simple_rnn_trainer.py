import numpy as np

class SimpleRNNTrainer:
    """
    A class to train a Simple RNN on a given dataset.

    Attributes:
        rnn (SimpleRNN): The RNN model to train.
        epochs (int): The number of training epochs.
    """

    def __init__(self, rnn, epochs):
        """
        Initializes the trainer with the RNN model and number of epochs.

        Args:
            rnn (SimpleRNN): An instance of the SimpleRNN model.
            epochs (int): The number of training epochs.
        """
        self.rnn = rnn
        self.epochs = epochs

    def train(self, X, Y):
        """
        Trains the RNN on the provided input (X) and target (Y) sequences.

        Args:
            X (np.ndarray): Input sequence data of shape (sequence_length, input_size).
            Y (np.ndarray): Target sequence data of shape (sequence_length, output_size).
        """
        for epoch in range(self.epochs):
            total_loss = 0
            self.rnn.reset_hidden_state()  # Reset hidden state at the start of each epoch

            # Forward pass
            h_seq, y_pred_seq = self.rnn.forward(X)

            # Calculate loss (cross-entropy)
            for t in range(len(X)):
                y_true = Y[t].reshape(-1, 1)
                y_pred = y_pred_seq[t].reshape(-1, 1)
                loss = -np.sum(y_true * np.log(y_pred + 1e-8))  # Add a small epsilon for numerical stability
                total_loss += loss

            # Backward pass and parameter updates
            dW_xh, dW_hh, dW_hy, db_h, db_y = self.rnn.backward(X, Y, y_pred_seq)

            # Update parameters
            self.rnn.W_xh -= self.rnn.learning_rate * dW_xh
            self.rnn.W_hh -= self.rnn.learning_rate * dW_hh
            self.rnn.W_hy -= self.rnn.learning_rate * dW_hy
            self.rnn.b_h -= self.rnn.learning_rate * db_h
            self.rnn.b_y -= self.rnn.learning_rate * db_y

            # Print total loss for the epoch
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
