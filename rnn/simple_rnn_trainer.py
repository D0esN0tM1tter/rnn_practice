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
        Step 1: Initialize the trainer.
        - Bind the RNN model and the number of epochs to train.
        
        Args:
            rnn (SimpleRNN): An instance of the SimpleRNN model.
            epochs (int): The number of training epochs.
        """
        self.rnn = rnn
        self.epochs = epochs

    def train(self, X, Y):
        """
        Step 2: Train the RNN on the provided dataset.
        - Perform multiple epochs of forward and backward passes to update model weights.
        
        Args:
            X (np.ndarray): Input sequence data of shape (sequence_length, input_size).
            Y (np.ndarray): Target sequence data of shape (sequence_length, output_size).
        """
        for epoch in range(self.epochs):  # Loop through each epoch
            # Step 3: Initialize the loss for this epoch
            total_loss = 0

            # Step 4: Reset the RNN's hidden state before starting a new epoch
            self.rnn.reset_hidden_state()

            # Step 5: Forward pass
            # Compute hidden states and predictions for the input sequence
            h_seq, y_pred_seq = self.rnn.forward(X)

            # Step 6: Compute loss for the entire sequence
            # Cross-entropy loss is calculated for each time step
            for t in range(len(X)):  # Iterate through the sequence length
                y_true = Y[t].reshape(-1, 1)  # Get the true label for time step t
                y_pred = y_pred_seq[t].reshape(-1, 1)  # Get the predicted probabilities
                # Calculate cross-entropy loss and accumulate it
                loss = -np.sum(y_true * np.log(y_pred + 1e-8))  # Add epsilon for numerical stability
                total_loss += loss

            # Step 7: Backward pass
            # Calculate gradients for all parameters using backpropagation through time
            dW_xh, dW_hh, dW_hy, db_h, db_y = self.rnn.backward(X, Y, y_pred_seq)

            # Step 8: Update model parameters
            # Adjust the weights and biases using the computed gradients and learning rate
            self.rnn.W_xh -= self.rnn.learning_rate * dW_xh
            self.rnn.W_hh -= self.rnn.learning_rate * dW_hh
            self.rnn.W_hy -= self.rnn.learning_rate * dW_hy
            self.rnn.b_h -= self.rnn.learning_rate * db_h
            self.rnn.b_y -= self.rnn.learning_rate * db_y

            # Step 9: Output training progress
            # Print the total loss for the current epoch to track training
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
