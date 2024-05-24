import numpy as np  
import matplotlib.pyplot as plt  

# Set a random seed for reproducibility (same results when running multiple times)
np.random.seed(10)

# Generate random X values between 1 and 20 (inclusive) with a size of 20
X = np.random.randint(1, 20 + 1, size=20)

# Generate Y values based on the equation Y = 2.3x + 3 with added random noise
Y = 2.3 * X + 3 + np.random.randn(20)

# Initialize the slope (m) and y-intercept (c) with random values
m, c = np.random.random(), np.random.random()

# Set the number of epochs (iterations) for gradient descent
epoch = 1000

# Set the learning rate for gradient descent
lr = 0.01

# Initialize empty lists to store mean squared error (mse) and epochs (ep)
mse = []  # List to store mean squared error for each epoch
ep = []  # List to store epoch number (iteration number)

# Initialize empty lists to store m and c values during each epoch
m_values = []
c_values = []

# Perform gradient descent for the specified number of epochs
for e in range(epoch):
    # Predict Y values based on current slope (m) and y-intercept (c)
    Y_pred = m * X + c

    # Calculate the error between predicted and actual Y values
    error = Y_pred - Y

    # Calculate the mean squared error (MSE)
    mse_value = np.sum(error ** 2) / (2 * len(Y_pred))
    mse.append(mse_value)  # Append MSE for this epoch

    # Append epoch number (iteration number)
    ep.append(e + 1)

    # Store current values of slope (m) and y-intercept (c)
    m_values.append(m)
    c_values.append(c)

    # Calculate gradients for slope (m) and y-intercept (c) using partial derivatives
    gradient_c = np.sum(error) / len(Y_pred)
    gradient_m = np.sum(error * (X)) / len(Y_pred)

    # Update slope (m) and y-intercept (c) using gradient descent with learning rate
    m = m - gradient_m * lr
    c = c - gradient_c * lr

# Print the final fitted equation based on the learned slope (m) and y-intercept (c)
print(f"Eqn: Y = {m:.4f}x + {c:.4f}")

# Create a figure with 2 rows and 2 columns for subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# Subplot 1: X vs Y (actual data) and X vs Y_pred (fitted line)
axes[0, 0].scatter(X, Y, label="Actual Data", color = 'b')
axes[0, 0].plot(X, Y_pred, color='r', label="Fitted Line")
axes[0, 0].set_xlabel("X")
axes[0, 0].set_ylabel("Y")
axes[0, 0].set_title("Data and Fitted Line")
axes[0, 0].legend()

# Subplot 2: Epoch vs MSE (mean squared error over epochs)
axes[0, 1].plot(ep, mse, label="Mean Squared Error", color = 'b')
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("MSE")
axes[0, 1].set_title("Mean Squared Error over Epochs")
axes[0, 1].legend()

# Subplot 3: m_values vs MSE (slope values vs mean squared error)
axes[1, 0].plot(m_values, mse, label="MSE", color = 'r')
axes[1, 0].plot(m_values, mse, 'o', color = 'b')  # Add markers (dots) for better visualization
axes[1, 0].set_xlabel("m Values")
axes[1, 0].set_ylabel("MSE")
axes[1, 0].set_title("m Values vs MSE")
axes[1, 0].legend()

# Subplot 4: c_values vs MSE
axes[1, 1].plot(c_values, mse, label="MSE", color = 'r')
axes[1, 1].plot(c_values, mse, 'o', color = 'b')
axes[1, 1].set_xlabel("c Values")
axes[1, 1].set_ylabel("MSE")
axes[1, 1].set_title("c Values vs MSE")
axes[1, 1].legend()

# Tight layout for better spacing
plt.tight_layout()

plt.show()