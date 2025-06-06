import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal_module  

def evaluate_signal(signal_params, t):
    """Evaluate the continuous-time signal at given time points t."""
    raw_signal = np.zeros_like(t)  
    for echo in signal_params['echoes']:
        strength = echo['strength']
        t_delay = echo['t_delay']
        duration = echo['duration']
        f_ultrasound = echo['f_ultrasound']
        decay = echo['decay']
        envelope = np.exp(-(t - t_delay) / decay) * (t >= t_delay) * (t - t_delay < duration)
        raw_signal += strength * np.sin(2 * np.pi * f_ultrasound * (t - t_delay)) * envelope

    b, a = signal_params['filter_coeffs']
    filtered = signal_module.filtfilt(b, a, raw_signal)  
    return filtered

def greedy_energy_based_channel_subsampling(
    signal_params,
    num_select,
    fs,
    sensor_angles
):
    num_sensors = len(signal_params)

    # Ensure num_select is an integer
    num_select = int(num_select)


    durations = [params['duration'] for params in signal_params]
    max_duration = max(durations)
    N_temp = int(max_duration * fs)
    t_temp = np.linspace(0, max_duration, N_temp)

    # Energy of each channel
    channel_energy = np.zeros(num_sensors)
    temp_signals = np.zeros((num_sensors, N_temp))
    for i in range(num_sensors):
        signal_params[i]['duration'] = max_duration  
        temp_signal = evaluate_signal(signal_params[i], t_temp)
        temp_signals[i] = temp_signal
        channel_energy[i] = np.sum(temp_signal**2)

    # Selecting sensors with highest energy
    selected_indices = np.argsort(channel_energy)[-num_select:][::-1]
    selected_params = [signal_params[i] for i in selected_indices]


    index_ops = num_sensors
    data_ops = num_select * N_temp
    total_ops = index_ops + data_ops

    processing_time = max_duration
    print(f"Greedy-selected channels: {selected_indices.tolist()}")
    print(f"Total Operations: {total_ops}")


    rows = int(np.ceil(num_select / 2))
    cols = min(num_select, 2)
    plt.figure(figsize=(12, 4 * rows))
    for i, sensor_idx in enumerate(selected_indices):
        plt.subplot(rows, cols, i + 1)
        plt.plot(t_temp * 1e6, temp_signals[sensor_idx] * 1e3, label='Selected', alpha=0.8)
        plt.title(f'Sensor {sensor_idx+1} (Angle: {np.rad2deg(sensor_angles[sensor_idx]):.1f}°)')
        plt.xlabel('Time (µs)')
        plt.ylabel('Voltage (mV)')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        'subsampled_signal_params': selected_params,
        'selected_indices': selected_indices.tolist(),
        'Total Operations': total_ops,
    }