import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    return -2 * (y_true - y_pred)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(0)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_hidden = np.random.rand(hidden_size)
    bias_output = np.random.rand(output_size)
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def forward_pass(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)
    return hidden_layer_output, output_layer_output

def backward_pass(X, y, hidden_layer_output, output_layer_output, weights_hidden_output, learning_rate):
    output_layer_error = mean_squared_error_derivative(y, output_layer_output) * sigmoid_derivative(output_layer_output)
    hidden_layer_error = np.dot(output_layer_error, weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output_update = np.dot(hidden_layer_output.T, output_layer_error)
    bias_output_update = np.sum(output_layer_error, axis=0)
    
    weights_input_hidden_update = np.dot(X.T, hidden_layer_error)
    bias_hidden_update = np.sum(hidden_layer_error, axis=0)
    
    return weights_hidden_output_update, bias_output_update, weights_input_hidden_update, bias_hidden_update

def update_parameters(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, 
                      weights_input_hidden_update, weights_hidden_output_update, 
                      bias_hidden_update, bias_output_update, learning_rate):
    weights_hidden_output -= learning_rate * weights_hidden_output_update
    bias_output -= learning_rate * bias_output_update
    weights_input_hidden -= learning_rate * weights_input_hidden_update
    bias_hidden -= learning_rate * bias_hidden_update
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def train(X, y, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = forward_pass(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
        loss = mean_squared_error(y, output_layer_output)
        weights_hidden_output_update, bias_output_update, weights_input_hidden_update, bias_hidden_update = backward_pass(X, y, hidden_layer_output, output_layer_output, weights_hidden_output, learning_rate)
        weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = update_parameters(
            weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, 
            weights_input_hidden_update, weights_hidden_output_update, 
            bias_hidden_update, bias_output_update, learning_rate
        )
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def predict(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    _, output_layer_output = forward_pass(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    return output_layer_output

input_size = 2
hidden_size = 2
output_size = 1
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train(X, y, input_size, hidden_size, output_size)

print("Training complete.")

predictions = predict(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
print("Predictions:")
print(predictions)
