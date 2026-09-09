"""Run a three-layer neural network forward with NumPy."""

import numpy as np


def sigmoid(values):
    values = np.clip(values, -50, 50)
    return 1 / (1 + np.exp(-values))


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


rng = np.random.default_rng(0)
network = {
    "W1": rng.normal(0, 0.3, size=(2, 4)),
    "b1": np.zeros(4),
    "W2": rng.normal(0, 0.3, size=(4, 3)),
    "b2": np.zeros(3),
    "W3": rng.normal(0, 0.3, size=(3, 3)),
    "b3": np.zeros(3),
}
features = np.array([[0.2, 0.8], [0.9, 0.1]])

hidden = sigmoid(features @ network["W1"] + network["b1"])
hidden = np.maximum(0, hidden @ network["W2"] + network["b2"])
logits = hidden @ network["W3"] + network["b3"]
probabilities = softmax(logits)

print("input:", features.shape)
print("logits:", logits.shape)
print("probabilities:\n", probabilities)
print("predicted classes:", probabilities.argmax(axis=1))
