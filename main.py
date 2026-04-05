import numpy as np
import pandas as pd

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("Loading Data...")
data = pd.read_csv("train.csv").values
np.random.shuffle(data)  # Shuffle for better training

# Split into Validation (first 1000) and Training
data_test = data[0:1000]
x_test = data_test[:, 1:] / 255.0  # NORMALIZE pixels to 0-1
y_test = data_test[:, 0]

data_train = data[1000:]
x_train = data_train[:, 1:] / 255.0  # NORMALIZE pixels to 0-1
y_train = data_train[:, 0]

print(f"Training Data Shape: {x_train.shape}")


# ==========================================
# 2. RAW MATH FUNCTIONS
# ==========================================
def one_hot(Y, num_classes=10):
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_gradient(Z):
    s = sigmoid(Z)
    return s * (1 - s)


def softmax(Z):
    # Subtracting max(Z) prevents math overflow errors
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def prepend(X):
    # Adding a column of 1s to handle the Bias weight efficiently
    return np.insert(X, 0, 1, axis=1)


def weights_init(n_input, n_hidden, n_classes):
    np.random.seed(0)
    # W1 shape: (785, 100) -> 784 inputs + 1 bias
    w1 = np.random.randn(n_input + 1, n_hidden) * np.sqrt(1.0 / (n_input + 1))
    # W2 shape: (101, 10) -> 100 hidden + 1 bias
    w2 = np.random.randn(n_hidden + 1, n_classes) * np.sqrt(1.0 / (n_hidden + 1))
    return w1, w2


# ==========================================
# 3. FORWARD & BACKWARD PROPAGATION
# ==========================================
def forward(X, w1, w2):
    Z1 = np.matmul(prepend(X), w1)
    H = sigmoid(Z1)
    Z2 = np.matmul(prepend(H), w2)
    Y_hat = softmax(Z2)
    return Z1, H, Z2, Y_hat


def backprop(X, Y_one_hot, Z1, H, Y_hat, w2):
    m = X.shape[0]

    # Layer 2 gradients
    dZ2 = Y_hat - Y_one_hot
    w2_gradient = np.matmul(prepend(H).T, dZ2) / m

    # Layer 1 gradients (ignore the bias row of w2 for backprop)
    w2_no_bias = w2[1:, :]
    dZ1 = np.matmul(dZ2, w2_no_bias.T) * sigmoid_gradient(Z1)
    w1_gradient = np.matmul(prepend(X).T, dZ1) / m

    return w1_gradient, w2_gradient


# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def get_predictions(Y_hat):
    return np.argmax(Y_hat, axis=1)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size * 100


def train_model(x_train, y_train, x_test, y_test, n_hidden_nodes=100, epochs=50, batch_size=64, lr=0.1):
    n_input = x_train.shape[1]
    n_classes = 10
    w1, w2 = weights_init(n_input, n_hidden_nodes, n_classes)
    m = x_train.shape[0]

    print("Starting Training...")
    for epoch in range(epochs):
        # Shuffle batches
        permutation = np.random.permutation(m)
        x_shuffled = x_train[permutation]
        y_shuffled = y_train[permutation]

        for i in range(0, m, batch_size):
            x_batch = x_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            y_batch_one_hot = one_hot(y_batch, n_classes)

            Z1, H, Z2, Y_hat = forward(x_batch, w1, w2)
            w1_grad, w2_grad = backprop(x_batch, y_batch_one_hot, Z1, H, Y_hat, w2)

            w1 -= lr * w1_grad
            w2 -= lr * w2_grad

        # Report Accuracy
        if epoch % 5 == 0 or epoch == epochs - 1:
            _, _, _, Y_hat_test = forward(x_test, w1, w2)
            accuracy = get_accuracy(get_predictions(Y_hat_test), y_test)
            print(f"Epoch {epoch:2d} > Validation Accuracy: {accuracy:.2f}%")

    return w1, w2


# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    w1_final, w2_final = train_model(x_train, y_train, x_test, y_test, n_hidden_nodes=128, epochs=100, batch_size=64,
                                     lr=0.15)
    print("Training Complete. Project Panthu13147 is a Success!")