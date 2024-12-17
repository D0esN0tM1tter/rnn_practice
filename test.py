import numpy as np
from simple_rnn import SimpleRNN
from simple_rnn_trainer import SimpleRNNTrainer

# One-hot encoded input and output sequences
X = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
Y = [np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])]

# Initialize the RNN
input_size = 3
hidden_size = 5
output_size = 3
learning_rate = 0.1
epochs = 1000

rnn = SimpleRNN(input_size, hidden_size, output_size, learning_rate)
trainer = SimpleRNNTrainer(rnn, epochs)
trainer.train(X, Y)
print('\n-----training ended successfully ------\n')

# Test the model
print("Testing the trained RNN:")
for i, x_t in enumerate(X):
    _, y_pred = rnn.forward(x_t)  # Perform forward pass on each test input
    predicted_index = np.argmax(y_pred)  # Get the index of the max probability
    print(f"Test Input {i + 1}: {x_t.ravel()} -> Predicted Output Index: {predicted_index}, Probabilities: {y_pred.ravel()}")
