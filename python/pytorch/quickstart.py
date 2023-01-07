#!/usr/bin/env python3
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchviz import make_dot
import pdb

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

# classes in dataset
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def main():
    print(f"Using {device} device")

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    #   each element in the dataloader iterable will return a batch of 64 features and labels.
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    fname = "model.pth"
    if not os.path.exists(fname):
        model = NeuralNetwork().to(device)
    else:
        model = NeuralNetwork()
        model.load_state_dict(torch.load(fname))
        print("reloaded model from file '{fname}'")

    print(model)
    # make_dot(model.linear_relu_stack).render("attached", format="png")

    infererence(test_data, model)
    exit(0)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    epochs = 5
    for t in range(epochs):
        print(f"\n***Epoch {t+1}/{epochs}***")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

        torch.save(model.state_dict(), fname)
        print(f"Saved PyTorch Model State to '{fname}")
    print("done!")


# Define model
class NeuralNetwork(nn.Module):
    """
    Every module in pytorch subclasses nn.Module
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html

    See also self.set_extra_state for handling (optional) extra state when calling load_state_dict()
    """

    def __init__(self):
        super().__init__()
        # the flatten layer converts each 28*28 image into "a contiguous array of 784 pixel values"
        #   (see practice.ipynb for demo)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # input layer has 28*28=784 inputs, and 512 outputs
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """
        Defines the forward computation at every call.
        Note: This shouldn't be called directly.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model: NeuralNetwork, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # place module in training mode (effect e.g. dropout depends on type of module)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model: NeuralNetwork, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # set module in evaluation mode (same as model.train(False))
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def infererence(test_data, model: NeuralNetwork):
    model.eval()
    with torch.no_grad():
        for i in range(10):
            x, y = test_data[i][0], test_data[i][1]
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f"predicted: {predicted}, actual: {actual}")


if __name__ == "__main__":
    main()
