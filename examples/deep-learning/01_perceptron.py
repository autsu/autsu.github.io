"""Build logic gates from perceptrons."""

from itertools import product


def perceptron(x1, x2, w1, w2, bias):
    weighted_sum = x1 * w1 + x2 * w2 + bias
    return int(weighted_sum > 0)


def xor_gate(x1, x2):
    # 组合两个第一层感知机，得到单层感知机无法表示的 XOR。
    or_value = perceptron(x1, x2, 0.5, 0.5, -0.2)
    and_value = perceptron(x1, x2, 0.5, 0.5, -0.7)
    return perceptron(or_value, and_value, 0.5, -0.5, -0.2)


for name, weights, bias in [
    ("AND", (0.5, 0.5), -0.7),
    ("OR", (0.5, 0.5), -0.2),
    ("NAND", (-0.5, -0.5), 0.7),
]:
    print(name)
    for x1, x2 in product([0, 1], repeat=2):
        print((x1, x2), "->", perceptron(x1, x2, *weights, bias))

print("XOR")
for x1, x2 in product([0, 1], repeat=2):
    print((x1, x2), "->", xor_gate(x1, x2))
