"""Train a two-layer classifier with hand-written backpropagation."""

import numpy as np


def make_data(seed=1):
    rng = np.random.default_rng(seed)
    left = rng.normal([-1.0, 0.0], 0.35, size=(100, 2))
    right = rng.normal([1.0, 0.0], 0.35, size=(100, 2))
    features = np.concatenate([left, right])
    labels = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    order = rng.permutation(len(labels))
    return features[order], labels[order]


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


rng = np.random.default_rng(0)
features, labels = make_data()
W1 = rng.normal(0, 0.2, size=(2, 8))
b1 = np.zeros(8)
W2 = rng.normal(0, 0.2, size=(8, 2))
b2 = np.zeros(2)

for step in range(200):
    hidden_linear = features @ W1 + b1
    hidden = np.maximum(0, hidden_linear)
    logits = hidden @ W2 + b2
    probabilities = softmax(logits)
    loss = -np.mean(np.log(probabilities[np.arange(len(labels)), labels] + 1e-7))

    dlogits = probabilities.copy()
    dlogits[np.arange(len(labels)), labels] -= 1
    dlogits /= len(labels)
    dW2 = hidden.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dhidden = dlogits @ W2.T
    dhidden[hidden_linear <= 0] = 0
    dW1 = features.T @ dhidden
    db1 = dhidden.sum(axis=0)

    W1 -= 0.5 * dW1
    b1 -= 0.5 * db1
    W2 -= 0.5 * dW2
    b2 -= 0.5 * db2

    if (step + 1) % 40 == 0:
        print(f"step={step + 1:03d}, loss={loss:.4f}")

print("accuracy:", np.mean(logits.argmax(axis=1) == labels))
