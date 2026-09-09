"""Python/NumPy warm-up: shapes, broadcasting and matrix multiplication."""

import numpy as np


features = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
weights = np.array([0.5, -0.2])
bias = 1.0

print("features shape:", features.shape)
print("first sample:", features[0])
scores = features @ weights + bias
print("scores:", scores)
targets = np.array([2.0, 5.0, 8.0])
print("mean squared error:", np.mean((scores - targets) ** 2))

offset = np.array([10.0, 20.0])
print("broadcast result:\n", features + offset)

# 画一条候选直线，确认输入与预测值的关系。
try:
    import matplotlib.pyplot as plt

    x = np.linspace(0, 3, 100)
    plt.scatter([0, 1, 2, 3], [2, 5, 8, 11], label="samples")
    plt.plot(x, 3 * x + 2, label="y = 3x + 2")
    plt.legend()
    plt.show()
except ImportError:
    print("matplotlib is not installed; skip the plot")
