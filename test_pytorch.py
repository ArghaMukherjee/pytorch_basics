import torch

# Define the data
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).view(-1, 1)
y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32).view(-1, 1)

# Define the model
model = torch.nn.Linear(1, 1)

# Define the loss function
loss_fn = torch.nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the model
y_pred = model(x)
print(y_pred)
