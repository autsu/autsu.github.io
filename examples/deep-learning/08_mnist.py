"""Train and evaluate a small CNN on MNIST."""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_items += images.size(0)
    return total_loss / total_items, total_correct / total_items


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
full_train = datasets.MNIST("data", train=True, download=True, transform=transform)
train_set, validation_set = random_split(
    full_train, [55_000, 5_000], generator=torch.Generator().manual_seed(0)
)
test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=256)
test_loader = DataLoader(test_set, batch_size=256)

model = MnistCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_state = None
best_loss = float("inf")

for epoch in range(3):
    train_loss, train_accuracy = run_epoch(
        model, train_loader, criterion, device, optimizer
    )
    validation_loss, validation_accuracy = run_epoch(
        model, validation_loader, criterion, device
    )
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_state = copy.deepcopy(model.state_dict())
    print(
        f"epoch={epoch + 1}, train_loss={train_loss:.4f}, "
        f"train_accuracy={train_accuracy:.3f}, "
        f"validation_accuracy={validation_accuracy:.3f}"
    )

if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), "mnist-cnn.pt")
test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
print(f"test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.3f}")
