import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (y = 2x + 3 + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 2 * X + 3 + np.random.randn(100, 1)

def gradient_descent(X, y, lr=0.1, n_iterations=100):
    m = 0  # slope
    c = 0  # intercept
    N = len(y)

    cost_history = []
    for i in range(n_iterations):
        y_pred = m * X + c
        error = y_pred - y

        # Compute gradients
        dm = (2/N) * np.sum(error * X)
        dc = (2/N) * np.sum(error)

        # Update parameters
        m -= lr * dm
        c -= lr * dc

        cost = (1/N) * np.sum((y_pred - y)**2)
        cost_history.append(cost)

        if i % 10 == 0:
            print(f"Iteration {i}: m = {m:.4f}, c = {c:.4f}, cost = {cost:.4f}")
    
    return m, c, cost_history

# Run Gradient Descent
m_opt, c_opt, cost_history = gradient_descent(X, y)

# Final Equation
print(f"\nOptimal Slope (m): {m_opt:.4f}")
print(f"Optimal Intercept (c): {c_opt:.4f}")

# Plot Data and Best Fit Line
plt.scatter(X, y, label="Data Points", color="blue")
plt.plot(X, m_opt * X + c_opt, color='red', label="Best Fit Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

# Plot Cost vs Iterations
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()
