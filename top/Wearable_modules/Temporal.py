import numpy as np
import matplotlib.pyplot as plt
import time


def temporal_subsampling_digital(adc_digital_data, fs, adc_bits=10, subsample_factor=4):
    duration = len(adc_digital_data) / fs
    t = np.linspace(0, duration * 1e6, len(adc_digital_data))  


    start_time = time.perf_counter()
    adc_subsampled = adc_digital_data[::subsample_factor]
    t_sub = t[::subsample_factor]
    end_time = time.perf_counter()
    subsampling_time = end_time - start_time  

  
    plt.figure(figsize=(12, 4))

    plt.subplot(2, 1, 1)
    plt.step(t, adc_digital_data, where='mid', color='blue', label='Original Digital Signal')
    plt.title(f"Original ADC Output (ADC Bits: {adc_bits})")
    plt.xlabel("Time (µs)")
    plt.ylabel("Digital Value")
    plt.grid(True)

 
    plt.subplot(2, 1, 2)
    plt.step(t_sub, adc_subsampled, where='mid', color='green', label='Subsampled Digital Signal')
    plt.title(f"Subsampled Digital Signal (Factor: {subsample_factor})")
    plt.xlabel("Time (µs)")
    plt.ylabel("Digital Value")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    print("=== Temporal Subsampling Summary ===")
    print(f"ADC Resolution      : {adc_bits} bits")
    print(f"Original Samples    : {len(adc_digital_data)}")
    print(f"Subsampled Samples  : {len(adc_subsampled)}")
    print(f"Subsampling Factor  : {subsample_factor}")
    print(f"Subsampling Time    : {subsampling_time*1e6:.2f} µs")
    print("====================================")

    return {
        "original_samples": len(adc_digital_data),
        "subsampled_samples": len(adc_subsampled),
        "subsampled_signal": adc_subsampled,
        "subsampling_time_us": subsampling_time * 1e6 
    }
