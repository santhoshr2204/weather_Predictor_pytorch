import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X, y):
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X)
        loss = F.binary_cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return loss.item()
