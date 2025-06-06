import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def evaluate_signal(signal_params, t):
    """Evaluate the continuous-time signal at given time points t."""
    signal = np.zeros_like(t)
    for echo in signal_params['echoes']:
        strength = echo['strength']
        t_delay = echo['t_delay']
        duration = echo['duration']
        f_ultrasound = echo['f_ultrasound']
        decay = echo['decay']
        envelope = np.exp(-(t - t_delay) / decay) * (t >= t_delay) * (t - t_delay < duration)
        signal += strength * np.sin(2 * np.pi * f_ultrasound * (t - t_delay)) * envelope
    return signal

def Sensors_outputs_analysis(data):
    # Load data
    sensor_signal_params = data["sensor_signal_params"]
    sensor_angles = data["sensor_angles"]
    num_sensors = data["num_sensors"]
    fs_sensor = data["fs_sensors"]
    lna_gain_db_sensor = data["lna_gain_db_sensor"]
    cutoff_freq_sensor = data["cutoff_freq_sensor"]
    R_load_sensor = data["R_load_sensor"]
    reflectors = data["reflectors"]
    probe_width_sensor = data["probe_width_sensor"]
    f_ultrasound_sensor = data["f_ultrasound_sensor"]
    transducer_sensitivity_sensor = data["transducer_sensitivity_sensor"]
    angular_range_sensor = data["angular_range_sensor"]
    reflectors_array = data["reflectors_array"]
    c_sensor = data["c_sensor"]

    lna_gain_db_sensor = 42.3  

    durations = [params['duration'] for params in sensor_signal_params]
    max_duration = max(durations)
    N_temp = int(max_duration * fs_sensor)
    t_temp = np.linspace(0, max_duration, N_temp)

    # Latency and power calculation parameters
    num_reflectors = len(reflectors)
    processor_speed_hz = 1e9  
    cycle_time_us = 1e-3 / processor_speed_hz  

    # Step 1: Signal Evaluation Latency and Power
    cycles_per_sample_per_reflector = 20
    total_cycles_signal_eval = N_temp * num_reflectors * cycles_per_sample_per_reflector * num_sensors
    time_signal_eval_us = total_cycles_signal_eval * cycle_time_us  
    power_signal_eval = 52  
    energy_signal_eval = (power_signal_eval * 1e-3) * (time_signal_eval_us * 1e-6)  

    # Signals for processing
    sensor_voltages = np.zeros((num_sensors, N_temp))
    for i in range(num_sensors):
        signal_params = sensor_signal_params[i]
        signal_params['duration'] = max_duration  
        sensor_voltages[i] = evaluate_signal(signal_params, t_temp) * transducer_sensitivity_sensor

    # Step 2: LNA Amplification Latency and Power
    time_lna_per_sensor_us = 1e-1  # 100 ns = 0.1 µs
    time_lna_total_us = time_lna_per_sensor_us * num_sensors
    power_lna_per_sensor = 0.1e-3  # mW (will be adjusted based on actual power calculation below)

    # LNA amplification
    lna_gain_linear = 10 ** (lna_gain_db_sensor / 20)
    amplified_signals = sensor_voltages * lna_gain_linear

    # Step 3: Low-Pass Filtering Latency and Power
    cycles_per_sample_filter = 50  # 4th-order filter, forward and backward
    total_cycles_filter = N_temp * cycles_per_sample_filter * num_sensors
    time_filter_us = total_cycles_filter * cycle_time_us
    power_filter = 42  # Adjusted to 42 mW to increase DSP power
    energy_filter = (power_filter * 1e-3) * (time_filter_us * 1e-6)

    # Low-pass filter
    nyquist = fs_sensor / 2
    normalized_cutoff = cutoff_freq_sensor / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
    filtered_signals = np.zeros_like(amplified_signals)
    for i in range(num_sensors):
        filtered_signals[i] = signal.filtfilt(b, a, amplified_signals[i])

    # Step 4: Power and Voltage Calculations Latency and Power
    cycles_per_sample_metrics = 10
    total_cycles_metrics = N_temp * cycles_per_sample_metrics * num_sensors * 3  # 3 stages
    time_metrics_us = total_cycles_metrics * cycle_time_us
    power_metrics = 31  # Adjusted to 31 mW to increase DSP power
    energy_metrics = (power_metrics * 1e-3) * (time_metrics_us * 1e-6)

    # Calculate power and voltage
    stages = ['Transducer', 'LNA', 'Filtered']
    results = {stage: {'V_peak': [], 'V_rms': [], 'Power': []} for stage in stages}

    for i in range(num_sensors):
        # Transducer stage
        v_trans = sensor_voltages[i]
        v_peak_trans = np.max(np.abs(v_trans))
        v_rms_trans = np.sqrt(np.mean(v_trans**2))
        power_trans = v_rms_trans**2 / R_load_sensor
        results['Transducer']['V_peak'].append(v_peak_trans)
        results['Transducer']['V_rms'].append(v_rms_trans)
        results['Transducer']['Power'].append(power_trans)

        # LNA stage
        v_lna = amplified_signals[i]
        v_peak_lna = np.max(np.abs(v_lna))
        v_rms_lna = np.sqrt(np.mean(v_lna**2))
        power_lna = v_rms_lna**2 / R_load_sensor
        results['LNA']['V_peak'].append(v_peak_lna)
        results['LNA']['V_rms'].append(v_rms_lna)
        results['LNA']['Power'].append(power_lna)

        # Filtered stage
        v_filt = filtered_signals[i]
        v_peak_filt = np.max(np.abs(v_filt))
        v_rms_filt = np.sqrt(np.mean(v_filt**2))
        power_filt = v_rms_filt**2 / R_load_sensor
        results['Filtered']['V_peak'].append(v_peak_filt)
        results['Filtered']['V_rms'].append(v_rms_filt)
        results['Filtered']['Power'].append(power_filt)

    avg_lna_power_uw = np.mean(results['LNA']['Power']) * 1e6  # Convert to µW
    avg_lna_power_mw = avg_lna_power_uw * 1e-3  # Convert to mW
    total_lna_power_mw = avg_lna_power_mw * num_sensors

    # Total Latency and Power
    total_latency_us = time_signal_eval_us + time_lna_total_us + time_filter_us + time_metrics_us
    total_power_signal_eval = power_signal_eval  # mW
    total_power_filter = power_filter  # mW
    total_power_metrics = power_metrics  # mW
    total_dsp_time_us = time_signal_eval_us + time_filter_us + time_metrics_us
    if total_dsp_time_us > 0:
        total_power_dsp = (
            (power_signal_eval * time_signal_eval_us +
             power_filter * time_filter_us +
             power_metrics * time_metrics_us) / total_dsp_time_us
        )
    else:
        total_power_dsp = 0
    total_power = total_power_dsp + total_lna_power_mw 

    rows = 8  
    cols = 4  
    plt.figure(figsize=(15, 20))  
    for i in range(num_sensors):
        plt.subplot(rows, cols, i + 1)
        plt.plot(t_temp * 1e6, filtered_signals[i] * 1e3)
        plt.title(f'Sensor {i + 1} (Angle: {np.rad2deg(sensor_angles[i]):.1f}°)')
        plt.xlabel('Time (us)')
        plt.ylabel('Voltage (mV)')
        plt.grid(True)
        if i // cols < rows - 1:
            plt.xlabel('')
        if i % cols != 0:
            plt.ylabel('')
    plt.tight_layout()
    plt.show()

    print("===============Ultrasound System Summary:===============")
    print(f"Number of Sensors          : {num_sensors}")
    print(f"Sampling Frequency         : {fs_sensor / 1e6:.1f} MHz")
    print(f"Ultrasound Frequency       : {f_ultrasound_sensor / 1e6:.1f} MHz")
    print(f"Transducer Sensitivity     : {transducer_sensitivity_sensor} V/Pa")
    print(f"LNA Gain                   : {lna_gain_db_sensor} dB ({lna_gain_linear:.1f}x)")
    print(f"Filter Cutoff Frequency    : {cutoff_freq_sensor / 1e6:.1f} MHz")
    print(f"Probe Width                : {probe_width_sensor * 1e3:.1f} mm")
    print(f"Angular Range              : ±{np.rad2deg(angular_range_sensor / 2):.1f}°")
    print("Reflectors:")
    for r in reflectors:
        print(f" - Depth: {r['depth'] * 1e3:.1f} mm, Lateral: {r['lateral'] * 1e3:.1f} mm, Strength: {r['strength']} Pa")

    print("\nSignal Metrics:")
    for stage in stages:
        print(f"\n{stage} Stage:")
        print(f"  Average Peak Voltage      : {np.mean(results[stage]['V_peak']) * 1e3:.3f} mV")
        print(f"  Average RMS Voltage       : {np.mean(results[stage]['V_rms']) * 1e3:.3f} mV")
        print(f"  Average Power             : {np.mean(results[stage]['Power']) * 1e6:.3f} uW")

    print("\nModule Metrics:")
    print(f"Total Latency             : {total_latency_us:.2f} µs")
    print(f"Total Power (DSP + LNA)   : {total_power:.2f} mW")
    print("---------------------------------------------------------")

    summary = {
        "num_sensors": num_sensors,
        "fs": fs_sensor,
        "f_ultrasound": f_ultrasound_sensor,
        "transducer_sensitivity": transducer_sensitivity_sensor,
        "lna_gain_db": lna_gain_db_sensor,
        "cutoff_freq": cutoff_freq_sensor,
        "probe_width": probe_width_sensor,
        "angular_range": angular_range_sensor,
        "sensor_analysis_latency": total_latency_us,
        "sensor_analysis_power_mw": total_power
    }

    processed_signal_params = []
    for i in range(num_sensors):
        orig_params = sensor_signal_params[i]
        processed_params = {
            'echoes': [
                {
                    'strength': echo['strength'] * lna_gain_linear,
                    't_delay': echo['t_delay'],
                    'duration': echo['duration'],
                    'f_ultrasound': echo['f_ultrasound'],
                    'decay': echo['decay']
                } for echo in orig_params['echoes']
            ],
            'duration': orig_params['duration'],
            'filter_coeffs': (b, a),
            'fs': fs_sensor
        }
        processed_signal_params.append(processed_params)

    return num_sensors, processed_signal_params, summary, sensor_angles