"""Minimal PyTorch SGD example: linear regression on synthetic data."""

import torch
from torch import nn
import matplotlib.pyplot as plt
import os

def main():
    torch.manual_seed(0)

    # Synthetic data: y = 3*x + noise
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), 1)
    y = 3 * x + 0.5 * torch.randn(x.size())

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    epochs = 100
    losses = []

    for epoch in range(epochs):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    params = list(model.parameters())
    print("Trained weight, bias:", params[0].item(), params[1].item())

    outdir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(outdir, exist_ok=True)
    try:
        plt.figure()
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training loss (SGD)')
        outpath = os.path.join(outdir, 'loss.png')
        plt.savefig(outpath)
        print(f"Saved loss plot to {outpath}")
    except Exception as e:
        print("Could not save plot:", e)

if __name__ == '__main__':
    main()
