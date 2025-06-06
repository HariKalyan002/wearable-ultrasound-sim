import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
import time

def compressive_sensing_digital(adc_digital_data, adc_bits=10, compression_ratio=0.25):
    N = len(adc_digital_data)                     # Original number of samples
    M = int(N * compression_ratio)                # Number of compressed measurements

    x = adc_digital_data.astype(np.float32)

    start_time = time.perf_counter()
    
    # Random sensing matrix Phi (M x N)
    np.random.seed(42)
    Phi = np.random.randn(M, N)
    
    # Compressed measurements: y = Phi * x
    y = Phi @ x
    compression_time = time.perf_counter() - start_time  

    # Measure processing time for reconstruction
    start_time = time.perf_counter()
    # Reconstruction using OMP
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=M)
    omp.fit(Phi, y)
    x_reconstructed = omp.coef_
    reconstruction_time = time.perf_counter() - start_time 

    # Total processing time
    total_processing_time = compression_time + reconstruction_time

    # Plot original digital data vs compressed signal
    plt.figure(figsize=(12, 5))

    # Original signal
    plt.subplot(2, 1, 1)
    plt.step(np.arange(N), adc_digital_data, where='mid', label='Original Digital ADC Output', color='blue')
    plt.title(f'Original Digital ADC Output (ADC Bits: {adc_bits})')
    plt.xlabel('Sample Index')
    plt.ylabel('Digital Value')
    plt.grid(True)

    # Compressed signal
    plt.subplot(2, 1, 2)
    plt.stem(np.arange(M), y, linefmt='g-', markerfmt='go', basefmt='k-', label='Compressed Measurements')
    plt.title(f'Compressed Digital Signal (Compression Ratio: {compression_ratio})')
    plt.xlabel('Compressed Sample Index')
    plt.ylabel('Compressed Value')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("=== Compressive Sensing Summary ===")
    print(f"ADC Resolution          : {adc_bits} bits")
    print(f"Original Samples        : {N}")
    print(f"Compressed Measurements : {M}")
    print(f"Compression Ratio       : {compression_ratio:.2f}")
    print(f"Compression Time        : {compression_time*1e6:.2f} µs")
    print(f"Reconstruction Time     : {reconstruction_time*1e6:.2f} µs")
    print(f"Total Processing Time   : {total_processing_time*1e6:.2f} µs")
    print("===================================")

    return {
        "original_samples": adc_digital_data,
        "compressed_signal": y,
        "reconstructed_signal": x_reconstructed,
        "compression_time_us": compression_time * 1e6, 
        "reconstruction_time_us": reconstruction_time * 1e6,  
        "total_processing_time_us": total_processing_time * 1e6  
    }