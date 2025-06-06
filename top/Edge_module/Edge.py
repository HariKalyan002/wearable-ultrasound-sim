import numpy as np
import matplotlib.pyplot as plt
from Wireless import Bluetooth, WiFi

def process_received_data(bit_stream, payloads, wireless_choice, distance=30, adc_bits=12, data_reduction_applied=False, decompression_at_server=False):

    received_bytes = [byte for payload in payloads for byte in payload]
    received_bits = ''.join(f'{byte:08b}' for byte in received_bytes)
    received_bits = received_bits[:len(bit_stream)]

    num_samples = len(bit_stream) // adc_bits

    # Computing RX power and latency using Wireless class
    if wireless_choice == 1:
        wireless_module = Bluetooth(distance=distance, data_stream=None, adc_bits=adc_bits, payload_bytes=len(payloads[0]) if payloads else 200)
    else:  # wireless_choice == 2
        wireless_module = WiFi(distance=distance, data_stream=None, adc_bits=adc_bits, payload_bytes=len(payloads[0]) if payloads else 200)

    rx_power_mw = wireless_module.Tx_power_noise(len(payloads[0]) if payloads else 200) * 1000  

    base_transmission_time = wireless_module.transmission_time(len(payloads[0]) if payloads else 200)  
    num_payloads = len(payloads) if payloads else 1
    total_bits = len(bit_stream)  # Total bits to transmit
    payload_size_bytes = len(payloads[0]) if payloads else 200
    total_data_bytes = payload_size_bytes * num_payloads

    typical_data_rate_mbps = 0.59  
    bits_per_payload = payload_size_bytes * 8
    total_transmission_time = base_transmission_time * num_payloads  # Base time for all payloads
    overhead_factor = (total_bits / (typical_data_rate_mbps * 1e6)) * 1e3 
    adjusted_transmission_time = total_transmission_time + (overhead_factor * 1e-3)  
    rx_latency_ms = adjusted_transmission_time * 1e3 


    base_processing_time_per_bit = 1e-6  
    k = 1.0  # Tuning ratio
    overhead_factor = 1 + k * (num_samples * (1 - 0.5)) / (num_samples * 0.5 + 1) if data_reduction_applied else 1.0
    processing_factor = base_processing_time_per_bit * overhead_factor  
    edge_processing_time_ms = len(bit_stream) * processing_factor * 1e3  

 
    reconstructed = None
    subsample_ratio = 0.5 if data_reduction_applied else 1.0
    sparse_reconstructed = None
    subsample_indices = None
    if not decompression_at_server:
        if data_reduction_applied:
            subsample_indices = np.random.choice(num_samples, size=int(num_samples * subsample_ratio), replace=False)
            subsample_indices.sort()
            sparse_reconstructed = [int(received_bits[i * adc_bits:(i + 1) * adc_bits], 2) 
                                 for i in subsample_indices if i * adc_bits < len(received_bits)]
            reconstructed = np.zeros(num_samples, dtype=int)
            reconstructed[subsample_indices] = sparse_reconstructed
            if len(subsample_indices) > 1:
                for i in range(len(subsample_indices) - 1):
                    start_idx = subsample_indices[i]
                    end_idx = subsample_indices[i + 1]
                    if end_idx > start_idx + 1:
                        num_points = end_idx - start_idx - 1
                        values = np.linspace(reconstructed[start_idx], reconstructed[end_idx], num_points + 2)[1:-1]
                        reconstructed[start_idx + 1:end_idx] = values.astype(int)
        else:
            reconstructed = [int(received_bits[i * adc_bits:(i + 1) * adc_bits], 2) 
                           for i in range(num_samples) if i * adc_bits < len(received_bits)]
            reconstructed = np.array(reconstructed)
            subsample_ratio = 1.0
            sparse_reconstructed = reconstructed
            subsample_indices = np.arange(len(reconstructed))

        # Plot 1: Reconstructed Signal (Sparse if data reduction applied)
        plt.figure(figsize=(12, 5))
        plt.plot(range(len(reconstructed)), reconstructed, label="Reconstructed Signal", linestyle='--', color='blue', marker='o')
        if data_reduction_applied:
            plt.plot(subsample_indices, sparse_reconstructed, 'ro', label="Subsampled Points", alpha=0.5)
        plt.title("Reconstructed Signal Samples at Edge Receiver")
        plt.xlabel("Sample Index")
        plt.ylabel("ADC Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot 2: Original vs Reconstructed Signal
        np.random.seed(42)
        original_signal = np.random.randint(0, 2**adc_bits, num_samples)
        plt.figure(figsize=(12, 5))
        plt.plot(range(num_samples), original_signal, label="Original Signal", color='green')
        plt.plot(range(len(reconstructed)), reconstructed, label="Reconstructed Signal", linestyle='--', color='red')
        plt.title("Original vs Reconstructed Signal at Edge Receiver")
        plt.xlabel("Sample Index")
        plt.ylabel("ADC Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    received_summary = {
        "RX Power (mW)": rx_power_mw,
        "RX Latency (ms)": rx_latency_ms,
        "Edge Decompression and Processing (ms)": edge_processing_time_ms,
        "Number of Payloads": len(payloads),
        "Bitstream Length (bits)": len(received_bits),
        "Bit Stream": bit_stream,  
        "Payloads": payloads,     
        "Subsample Ratio": subsample_ratio,
    }
    if not decompression_at_server:
        received_summary["Reconstructed Samples"] = len(reconstructed) if reconstructed is not None else 0
       
    return received_summary