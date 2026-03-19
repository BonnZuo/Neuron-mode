import numpy as np
import matplotlib.pyplot as plt

# ========== 补全 sigmoid 函数 ==========
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# =====================================

def relu(x):
    return np.maximum(0.0, x)

def tanh(x):
    return np.tanh(x)

# ========== 可修改参数 ==========
weights = np.array([0.8, -0.4, 0.6])
inputs = np.array([1.0, 0.5, 1.2])
bias = -0.2
# =================================

z = np.dot(inputs, weights) + bias

print("------ 人工神经元实验 ------")
print("输入:", inputs)
print("权重:", weights)
print("偏置:", bias)
print("加权和 z =", round(z, 3))
print("sigmoid(z) =", round(float(sigmoid(np.array([z]))[0]), 3))
print("ReLU(z) =", round(float(relu(np.array([z]))[0]), 3))
print("tanh(z) =", round(float(tanh(np.array([z]))[0]), 3))

# 绘图
x = np.linspace(-5, 5, 400)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x, sigmoid(x), color="tab:orange")
axes[0].set_title("Sigmoid")
axes[0].grid(alpha=0.3)

axes[1].plot(x, relu(x), color="tab:green")
axes[1].set_title("ReLU")
axes[1].grid(alpha=0.3)

axes[2].plot(x, tanh(x), color="tab:red")
axes[2].set_title("Tanh")
axes[2].grid(alpha=0.3)

for ax in axes:
    ax.set_xlabel("z")
    ax.set_ylabel("output")

fig.suptitle("不同激活函数的输出曲线", fontsize=14)
plt.tight_layout()
plt.show()