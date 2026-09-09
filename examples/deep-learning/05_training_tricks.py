"""Compare SGD, Momentum and Adam on a two-dimensional objective."""

import numpy as np


def optimize(name, learning_rate=0.2, steps=25):
    point = np.array([3.0, -2.0])
    velocity = np.zeros(2)
    first = np.zeros(2)
    second = np.zeros(2)
    history = []

    for step in range(1, steps + 1):
        gradient = np.array([point[0] / 10.0, 2.0 * point[1]])
        if name == "sgd":
            point -= learning_rate * gradient
        elif name == "momentum":
            velocity = 0.9 * velocity - learning_rate * gradient
            point += velocity
        else:
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient ** 2
            first_hat = first / (1 - 0.9 ** step)
            second_hat = second / (1 - 0.999 ** step)
            point -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        history.append(point[0] ** 2 / 20 + point[1] ** 2)
    return history


for name in ["sgd", "momentum", "adam"]:
    history = optimize(name)
    print(f"{name:8s}: {history[0]:.4f} -> {history[-1]:.6f}")

print("修改 learning_rate，观察不同更新方法的稳定性。")
