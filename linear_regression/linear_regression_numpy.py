import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv")

x = data["hours"].values
y = data["marks"].values

# Normalize (helps model learn better)
x = (x - np.mean(x)) / np.std(x)

# Initialize parameters
m = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(x)

# Training (Gradient Descent)
for i in range(epochs):
    y_pred = m * x + b

    dm = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    m = m - learning_rate * dm
    b = b - learning_rate * db

# Final prediction
y_pred = m * x + b

print("Slope (m):", m)
print("Intercept (b):", b)

# Sort for better line graph
idx = np.argsort(x)

# Plot
plt.scatter(x, y)
plt.plot(x[idx], y_pred[idx], color="red")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression using NumPy")
plt.show()
