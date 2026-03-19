import numpy as np

def mp_neuron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    output = 1 if weighted_sum >= threshold else 0
    return weighted_sum, output

# ========== 可修改参数 ==========
weights = np.array([0.9, 0.2, -0.5])
threshold = 1.0

inputs_a = np.array([1.0, 1.0, 0.0])
inputs_b = np.array([1.0, 0.0, 1.0])
# =================================

sum_a, output_a = mp_neuron(inputs_a, weights, threshold)
sum_b, output_b = mp_neuron(inputs_b, weights, threshold)

print("------ M-P 神经元实验 ------")
print("权重:", weights)
print("阈值:", threshold)
print()
print("样例1 输入:", inputs_a)
print("样例1 加权和:", round(sum_a, 3))
print("样例1 输出:", output_a)
print()
print("样例2 输入:", inputs_b)
print("样例2 加权和:", round(sum_b, 3))
print("样例2 输出:", output_b)