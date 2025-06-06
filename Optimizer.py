import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define parameters
i_options = [1, 2]  # 1 (lossless), 2 (lossy)
j_options = [1, 2, 3]  # 1 (Temporal Subsampling), 2 (Compressive Sensing), 3 (Beamforming)
dc_options = [1, 2]  # 1 (Edge decompression), 2 (Server decompression)
beta0_j = {1: 0.333, 2: 0.5, 3: 0.5}  # Minimum compression ratios
T_max = 50.0  # Maximum latency constraint (ms)
P_max = 300.0  # Maximum power constraint (mW)

# Define actual beta_j from results
beta_j_actual = {
    1: 0.3342,  # Temporal Subsampling: 1 - 0.6658
    2: 0.4994,  # Compressive Sensing: 1 - 0.5006
    3: 0.5006   # Beamforming: 1 - 0.4994
}

# Power and latency data from results
power_data = {
    1: 543.48,  # Temporal Subsampling
    2: 556.55,  # Compressive Sensing
    3: 556.54   # Beamforming
}

latency_data = {
    (1, 1): 15.7685,  # Temporal Subsampling, dc=1
    (1, 2): 25.1685,  # Temporal Subsampling, dc=2
    (2, 1): 18.8969,  # Compressive Sensing, dc=1
    (2, 2): 29.1269,  # Compressive Sensing, dc=2
    (3, 1): 26.5551,  # Beamforming, dc=1
    (3, 2): 44.0651   # Beamforming, dc=2
}

def power_wearable(beta_j, i, j):
    p_measured = power_data[j]
    beta_actual = beta_j_actual[j]
    beta0 = beta0_j[j]
    
    # Quadratic interpolation: p(beta_j) = a * (beta_j - beta0)^2 + p0
    # Assume p1 (at beta_j = 1) is higher; estimate p0 at beta0
    p1 = 600.0  # Assumed max power at beta_j = 1
    p0 = p_measured * 0.8  # Assume p0 is 80% of measured power at beta_actual
    a = (p1 - p0) / (1.0 - beta0)**2
    power = a * (beta_j - beta0)**2 + p0
    
    if i == 2:  # Lossy compression
        power *= 0.9
    
    return power

def latency(beta_j, i, j, dc):
    t_measured = latency_data[(j, dc)]
    beta_actual = beta_j_actual[j]
    beta0 = beta0_j[j]
    
    # Inverse interpolation: t(beta_j) = (a / beta_j) + b
    t1 = 50.0  # Assumed max latency at beta_j = 1
    t0 = t_measured * 0.5  # Assume t0 is 50% of measured latency at beta_actual
    a = (t1 - t0) * beta0 * 1.0 / (1.0 - beta0)
    b = t0 - (a / beta0)
    lat = (a / beta_j) + b

    if i == 2:  # Lossy compression
        lat *= 0.9
    
    return lat

def objective_power(x, i, j, dc):
    beta_j = x[0]
    return power_wearable(beta_j, i, j)

def constraint_latency(x, i, j, dc, T_max):
    beta_j = x[0]
    return T_max - latency(beta_j, i, j, dc)

def objective_latency(x, i, j, dc):
    beta_j = x[0]
    return latency(beta_j, i, j, dc)

def constraint_power(x, i, j, P_max):
    beta_j = x[0]
    return P_max - power_wearable(beta_j, i, j)

results_power = {}
results_latency = {}

for i in i_options:
    for j in j_options:
        for dc in dc_options:
            # Power optimization
            initial_beta = 0.7
            bounds = [(beta0_j[j], 1.0)]
            constraints = [{'type': 'ineq', 'fun': constraint_latency, 'args': (i, j, dc, T_max)}]
            result = minimize(objective_power, [initial_beta], args=(i, j, dc), bounds=bounds, constraints=constraints)
            beta_opt_power = result.x[0]
            p_wearable_opt = power_wearable(beta_opt_power, i, j)
            t_opt = latency(beta_opt_power, i, j, dc)
            results_power[(i, j, dc)] = {'beta': beta_opt_power, 'power': p_wearable_opt, 'latency': t_opt}

            # Latency optimization
            constraints = [{'type': 'ineq', 'fun': constraint_power, 'args': (i, j, P_max)}]
            result = minimize(objective_latency, [initial_beta], args=(i, j, dc), bounds=bounds, constraints=constraints)
            beta_opt_latency = result.x[0]
            t_opt = latency(beta_opt_latency, i, j, dc)
            p_wearable_opt = power_wearable(beta_opt_latency, i, j)
            results_latency[(i, j, dc)] = {'beta': beta_opt_latency, 'latency': t_opt, 'power': p_wearable_opt}

# Plot bar graphs
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Power optimization results
power_bars = []
power_labels = []
color_cycle = ['Blue', 'Orange', 'Green', 'Red', 'Purple', 'Brown']
for idx, ((i, j, dc), result) in enumerate(results_power.items()):
    label = f'i={i}, j={j}, DC={dc}, β={result["beta"]:.2f}'
    power_bars.append(result["power"])
    power_labels.append(label)
ax1.bar(range(len(power_bars)), power_bars, color=[color_cycle[idx % len(color_cycle)] for idx in range(len(power_bars))])
ax1.set_xlabel("Optimization Case")
ax1.set_ylabel("Power (mW)")
ax1.set_title("Power Optimization Results (Min P, T ≤ 50 ms)")
ax1.set_xticks(range(len(power_bars)))
ax1.set_xticklabels(power_labels, rotation=45, ha='right')
ax1.grid(True)

# Latency optimization results
latency_bars = []
latency_labels = []
for idx, ((i, j, dc), result) in enumerate(results_latency.items()):
    label = f'i={i}, j={j}, DC={dc}, β={result["beta"]:.2f}'
    latency_bars.append(result["latency"])
    latency_labels.append(label)
ax2.bar(range(len(latency_bars)), latency_bars, color=[color_cycle[idx % len(color_cycle)] for idx in range(len(latency_bars))])
ax2.set_xlabel("Optimization Case")
ax2.set_ylabel("Latency (ms)")
ax2.set_title("Latency Optimization Results (Min T, P ≤ 300 mW)")
ax2.set_xticks(range(len(latency_bars)))
ax2.set_xticklabels(latency_labels, rotation=45, ha='right')
ax2.grid(True)

plt.tight_layout()
plt.show()

print("Power Optimization Results (Min P, T ≤ 50 ms):")
for (i, j, dc), result in results_power.items():
    dc_str = "Server" if dc == 2 else "Edge"
    print(f"i={i}, j={j}, Decomp={dc_str}, β={result['beta']:.3f}, Power={result['power']:.2f} mW, Latency={result['latency']:.2f} ms")

print("\nLatency Optimization Results (Min T, P ≤ 300 mW):")
for (i, j, dc), result in results_latency.items():
    dc_str = "Server" if dc == 2 else "Edge"
    print(f"i={i}, j={j}, Decomp={dc_str}, β={result['beta']:.3f}, Latency={result['latency']:.2f} ms, Power={result['power']:.2f} mW")