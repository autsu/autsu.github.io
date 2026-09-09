"""Fit a noisy line with an explicit MSE gradient."""

import numpy as np


rng = np.random.default_rng(0)
x = np.linspace(-1, 1, 32)
y = 3.0 * x + 2.0 + 0.08 * rng.normal(size=x.shape)
weight = 0.0
bias = 0.0
learning_rate = 0.1

for step in range(300):
    prediction = weight * x + bias
    error = prediction - y
    loss = np.mean(error ** 2)

    # 这两个公式就是 MSE 对 weight 和 bias 的偏导数。
    dweight = np.mean(2.0 * error * x)
    dbias = np.mean(2.0 * error)
    weight -= learning_rate * dweight
    bias -= learning_rate * dbias

    if (step + 1) % 50 == 0:
        print(
            f"step={step + 1:03d}, loss={loss:.5f}, "
            f"weight={weight:.3f}, bias={bias:.3f}"
        )

print(f"learned line: y = {weight:.3f}x + {bias:.3f}")
