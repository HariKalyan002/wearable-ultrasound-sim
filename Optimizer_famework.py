import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define parameters
i_options = [1, 2]  # Compression scenarios: 1 (lossless), 2 (lossy)
j_options = [1, 2, 3]  # Data reduction techniques: 1 (subsampling), 2 (compressive sensing), 3 (beamforming)
beta0_j = {1: 0.3, 2: 0.5, 3: 0.4}  # Minimum compression ratios
T_max = 50.0  # Maximum latency constraint (ms)
P_max = 300.0  # Maximum power constraint (mW)

def power_wearable(beta_j, i, j):

    if j == 1:  # Temporal Subsampling

        p0, beta0 = 150.0, 0.333
    elif j == 2:  # Compressive Sensing

        p0, beta0 = 180.0, 0.50
    else:  # Beamforming (j == 3)

        p0, beta0 = 200.0, 0.50
    
    p1 = 250.0 
    
    # Quadratic interpolation (convex): p(beta_j) = a * (beta_j - beta0)^2 + p0
    a = (p1 - p0) / (1.0 - beta0)**2
    power = a * (beta_j - beta0)**2 + p0
    
    # Adjust for compression scenario (i)
    if i == 2:  # Lossy compression
        power *= 0.9  # 10% reduction in power
    
    return power

def latency(beta_j, i, j):
    if j == 1:  # Temporal Subsampling
        # At beta_j 
        t0, beta0 = 20.0, 0.333
    elif j == 2:  # Compressive Sensing
        t0, beta0 = 25.0, 0.50
    else:  # Beamforming
        t0, beta0 = 30.0, 0.50
    
    t1 = 40.0 
    
    # Inverse interpolation (convex): t(beta_j) = (a / beta_j) + b
    a = (t1 - t0) * beta0 * 1.0 / (1.0 - beta0)
    b = t0 - (a / beta0)
    lat = (a / beta_j) + b

    if i == 2:  # Lossy compression
        lat *= 0.9  
    
    return lat

def objective_power(x, i, j):
    beta_j = x[0]
    return power_wearable(beta_j, i, j)

def constraint_latency(x, i, j, T_max):
    beta_j = x[0]
    return T_max - latency(beta_j, i, j)

def objective_latency(x, i, j):
    beta_j = x[0]
    return latency(beta_j, i, j)

def constraint_power(x, i, j, P_max):
    beta_j = x[0]
    return P_max - power_wearable(beta_j, i, j)

results_power = {}
results_latency = {}

for i in i_options:
    for j in j_options:
        # Power optimization 
        initial_beta = 0.7
        bounds = [(beta0_j[j], 1.0)]
        constraints = [{'type': 'ineq', 'fun': constraint_latency, 'args': (i, j, T_max)}]
        result = minimize(objective_power, [initial_beta], args=(i, j), bounds=bounds, constraints=constraints)
        beta_opt_power = result.x[0]
        p_wearable_opt = power_wearable(beta_opt_power, i, j)
        t_opt = latency(beta_opt_power, i, j)
        results_power[(i, j)] = {'beta': beta_opt_power, 'power': p_wearable_opt, 'latency': t_opt}

        # Latency optimization 
        initial_beta = 0.7
        bounds = [(beta0_j[j], 1.0)]
        constraints = [{'type': 'ineq', 'fun': constraint_power, 'args': (i, j, P_max)}]
        result = minimize(objective_latency, [initial_beta], args=(i, j), bounds=bounds, constraints=constraints)
        beta_opt_latency = result.x[0]
        t_opt = latency(beta_opt_latency, i, j)
        p_wearable_opt = power_wearable(beta_opt_latency, i, j)
        results_latency[(i, j)] = {'beta': beta_opt_latency, 'latency': t_opt, 'power': p_wearable_opt}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Power optimization results
for (i, j), result in results_power.items():
    label = f'i={i}, j={j}, β={result["beta"]:.2f}'
    ax1.scatter(result["power"], result["latency"], label=label)
ax1.set_xlabel("Power (mW)")
ax1.set_ylabel("Latency (ms)")
ax1.set_title("Power Optimization (Min P, T ≤ 50 ms)")
ax1.legend()
ax1.grid(True)

# Latency optimization results
for (i, j), result in results_latency.items():
    label = f'i={i}, j={j}, β={result["beta"]:.2f}'
    ax2.scatter(result["power"], result["latency"], label=label)
ax2.set_xlabel("Power (mW)")
ax2.set_ylabel("Latency (ms)")
ax2.set_title("Latency Optimization (Min T, P ≤ 300 mW)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


print("Power Optimization Results (Min P, T ≤ 50 ms):")
for (i, j), result in results_power.items():
    print(f"i={i}, j={j}, β={result['beta']:.3f}, Power={result['power']:.2f} mW, Latency={result['latency']:.2f} ms")

print("\nLatency Optimization Results (Min T, P ≤ 300 mW):")
for (i, j), result in results_latency.items():
    print(f"i={i}, j={j}, β={result['beta']:.3f}, Latency={result['latency']:.2f} ms, Power={result['power']:.2f} mW")