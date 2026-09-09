"""Train a small PyTorch classifier on two groups of points."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(0)
generator = torch.Generator().manual_seed(7)
left = torch.randn(200, 2, generator=generator) * 0.45 + torch.tensor([-1.0, 0.0])
right = torch.randn(200, 2, generator=generator) * 0.45 + torch.tensor([1.0, 0.0])
features = torch.cat([left, right])
labels = torch.cat([
    torch.zeros(200, dtype=torch.long),
    torch.ones(200, dtype=torch.long),
])
loader = DataLoader(TensorDataset(features, labels), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(30):
    model.train()
    for batch_features, batch_labels in loader:
        optimizer.zero_grad()
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        accuracy = (logits.argmax(dim=1) == batch_labels).float().mean()
        print(f"epoch={epoch + 1:02d}, last_batch_accuracy={accuracy.item():.3f}")

model.eval()
with torch.no_grad():
    prediction = model(torch.tensor([[1.2, 0.1], [-1.1, 0.0]])).argmax(dim=1)
print("predicted classes:", prediction.tolist())
