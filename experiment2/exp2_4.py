import numpy as np
import matplotlib.pyplot as plt

def simulate_neuron(input_current, threshold, decay=0.85, T=50, reset_value=0.0):
    voltage = np.zeros(T)
    spikes = np.zeros(T)
    for t in range(1, T):
        if spikes[t-1] == 1:
            prev_v = reset_value
        else:
            prev_v = voltage[t-1]
        voltage[t] = decay * prev_v + input_current
        if voltage[t] >= threshold:
            spikes[t] = 1
    return voltage, spikes

def main():
    T = 50
    time = np.arange(T)

    # 情况1：相同阈值，不同电流
    threshold_same = 1.0
    current_small = 0.12
    current_large = 0.28

    voltage_small, spikes_small = simulate_neuron(current_small, threshold_same, T=T)
    voltage_large, spikes_large = simulate_neuron(current_large, threshold_same, T=T)

    # 情况2：相同电流，不同阈值
    input_current_same = 0.22
    threshold_low = 0.75
    threshold_high = 1.30

    voltage_low, spikes_low = simulate_neuron(input_current_same, threshold_low, T=T)
    voltage_high, spikes_high = simulate_neuron(input_current_same, threshold_high, T=T)

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axes[0,0].plot(time, voltage_small, label=f"I={current_small}", color="tab:blue")
    axes[0,0].plot(time, voltage_large, label=f"I={current_large}", color="tab:orange")
    axes[0,0].axhline(threshold_same, linestyle="--", color="green", label="threshold")
    axes[0,0].set_title("相同阈值下膜电位")
    axes[0,0].set_ylabel("voltage")
    axes[0,0].grid(alpha=0.3)
    axes[0,0].legend()

    axes[1,0].step(time, spikes_small, where="mid", label=f"I={current_small}", color="tab:blue")
    axes[1,0].step(time, spikes_large, where="mid", label=f"I={current_large}", color="tab:orange")
    axes[1,0].set_title("相同阈值下脉冲")
    axes[1,0].set_xlabel("time step")
    axes[1,0].set_ylabel("spike")
    axes[1,0].grid(alpha=0.3)
    axes[1,0].legend()

    axes[0,1].plot(time, voltage_low, label=f"th={threshold_low}", color="tab:red")
    axes[0,1].plot(time, voltage_high, label=f"th={threshold_high}", color="tab:purple")
    axes[0,1].axhline(threshold_low, linestyle="--", color="tab:red", alpha=0.6)
    axes[0,1].axhline(threshold_high, linestyle="--", color="tab:purple", alpha=0.6)
    axes[0,1].set_title("相同电流下膜电位")
    axes[0,1].set_ylabel("voltage")
    axes[0,1].grid(alpha=0.3)
    axes[0,1].legend()

    axes[1,1].step(time, spikes_low, where="mid", label=f"th={threshold_low}", color="tab:red")
    axes[1,1].step(time, spikes_high, where="mid", label=f"th={threshold_high}", color="tab:purple")
    axes[1,1].set_title("相同电流下脉冲")
    axes[1,1].set_xlabel("time step")
    axes[1,1].set_ylabel("spike")
    axes[1,1].grid(alpha=0.3)
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()