import numpy as np
import matplotlib.pyplot as plt

def add_wave(response, wave, start_time):
    for k, value in enumerate(wave):
        if start_time + k < len(response):
            response[start_time + k] += value

T = 20
time = np.arange(T)

# ========== 可修改参数 ==========
epsp = np.array([0.8, 0.4, 0.2])
ipsp = np.array([-0.7, -0.4, -0.2])
threshold = 1.0

exc_times = [2, 6, 7, 12, 13]
inh_times = [13]  # 可删掉观察变化
# =================================

response = np.zeros(T)
for t0 in exc_times:
    add_wave(response, epsp, t0)
for t0 in inh_times:
    add_wave(response, ipsp, t0)

exc_idx = np.array(exc_times)
inh_idx = np.array(inh_times)
output = (response >= threshold).astype(int)

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].stem(
    exc_idx, np.ones(len(exc_idx)),
    linefmt="tab:blue", markerfmt="bo", basefmt=" "
)
axes[0].stem(
    inh_idx, -np.ones(len(inh_idx)),
    linefmt="tab:red", markerfmt="ro", basefmt=" "
)
axes[0].set_title("输入事件时刻(蓝色兴奋，红色抑制)")
axes[0].set_ylabel("event")
axes[0].grid(alpha=0.3)

axes[1].plot(time, response, marker="o", label="总响应", color="tab:purple")
axes[1].axhline(threshold, color="green", linestyle="--", label="阈值")
axes[1].step(time, output, where="mid", label="二值输出", color="tab:orange")
axes[1].set_title("EPSP/IPSP 叠加响应")
axes[1].set_xlabel("time step")
axes[1].set_ylabel("response")
axes[1].set_xticks(np.arange(0, T, 2))
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()