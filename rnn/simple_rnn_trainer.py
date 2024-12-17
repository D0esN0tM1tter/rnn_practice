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
        """
        self.rnn = rnn
        self.epochs = epochs

    def train(self, X, Y):
        """
        Trains the RNN on the provided input (X) and target (Y) sequences.
        """
        for epoch in range(self.epochs):
            total_loss = 0
            self.rnn.reset_hidden_state()  # Reset hidden state at the start of each epoch

            for t in range(len(X)):
                x_t = X[t].reshape(-1, 1)
                y_true = Y[t].reshape(-1, 1)

                # Forward pass
                self.rnn.h_prev, y_t = self.rnn.forward(x_t)

                # Loss calculation (cross-entropy)
                loss = -np.sum(y_true * np.log(y_t))
                total_loss += loss

                # Backward pass and parameter updates
                dW_xh, dW_hh, dW_hy, db_h, db_y = self.rnn.backward(x_t, y_true)
                self.rnn.update_parameters(dW_xh, dW_hh, dW_hy, db_h, db_y)


            # Print total loss for the epoch
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
