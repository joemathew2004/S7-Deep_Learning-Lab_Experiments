import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

weights = torch.tensor([[40], [45], [50], [55], [60], [65], [70], [75], [80]], dtype=torch.float32)
noise = torch.empty(weights.size()).uniform_(-3, 3)
heights = 0.5 * weights + 100 + noise

# Normalize data (very important to avoid NaN with gradient descent)
mean_w = weights.mean()
std_w = weights.std()
weights_norm = (weights - mean_w) / std_w

mean_h = heights.mean()
std_h = heights.std()
heights_norm = (heights - mean_h) / std_h

# Step 2: Linear model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# Step 3: Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # safer learning rate

# Step 4: Training Loop
n_epochs = 1000
for epoch in range(n_epochs):
    outputs = model(weights_norm)
    loss = criterion(outputs, heights_norm)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Step 5: Get learned weights and bias in normalized space
[w, b] = model.parameters()
print(f"\nNormalized Slope (weight): {w.item():.4f}")
print(f"Normalized Intercept (bias): {b.item():.4f}")

# Step 6: Convert model back to original scale for plotting
# height = (w * (x - mean_w) / std_w + b) * std_h + mean_h
def denormalized_prediction(x):
    x_norm = (x - mean_w) / std_w
    y_norm = model(x_norm)
    return y_norm * std_h + mean_h

# Step 7: Plot results
predicted = denormalized_prediction(weights).detach()

plt.scatter(weights, heights, label='Noisy Data', color='blue')
plt.plot(weights, predicted, label='Fitted Line', color='red')
plt.xlabel("Weight (kg)")
plt.ylabel("Height (cm)")
plt.title("Simple Linear Regression with Noise (Normalized)")
plt.legend()
plt.show()

