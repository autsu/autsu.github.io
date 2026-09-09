import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_images(count, seed):
    # 随机生成横线和竖线两类灰度图片。
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(
        0,
        2,
        (count,),
        generator=generator,
    )
    # 图像 batch 的形状固定为 [N, C, H, W]。
    images = torch.zeros(count, 1, 16, 16)

    for index, label in enumerate(labels.tolist()):
        if label == 0:
            column = int(
                torch.randint(2, 14, (1,), generator=generator)
            )
            images[index, 0, :, column:column + 2] = 1.0
        else:
            row = int(
                torch.randint(2, 14, (1,), generator=generator)
            )
            images[index, 0, row:row + 2, :] = 1.0

    # 噪声让任务不再只是记忆固定位置的像素。
    noise = 0.12 * torch.randn(
        images.shape,
        generator=generator,
    )
    return (images + noise).clamp(0, 1), labels


class BarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 每次池化都让高和宽减半，通道数逐步增加。
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 2),
        )

    def forward(self, images):
        # 特征输出为 [N, 16, 4, 4]，随后展平给 Linear。
        features = self.features(images)
        return self.classifier(features)


torch.manual_seed(0)
# 不同 seed 生成互不相同的训练集和验证集。
train_images, train_labels = make_images(800, seed=10)
validation_images, validation_labels = make_images(200, seed=11)
train_loader = DataLoader(
    TensorDataset(train_images, train_labels),
    batch_size=32,
    shuffle=True,
)
validation_loader = DataLoader(
    TensorDataset(validation_images, validation_labels),
    batch_size=64,
)

model = BarClassifier()
criterion = nn.CrossEntropyLoss()
# logits 的形状是 [batch, 2]，标签的形状是 [batch]。
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

for epoch in range(8):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        # 一个训练 batch 的完整更新过程。
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # 验证只读取结果，不执行反向传播。
        for images, labels in validation_loader:
            logits = model(images)
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(
        f"epoch={epoch + 1}, "
        f"train_loss={total_loss / len(train_loader.dataset):.4f}, "
        f"validation_accuracy={correct / total:.3f}"
    )
