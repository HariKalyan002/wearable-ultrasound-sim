import numpy as np
import matplotlib.pyplot as plt
import time
from Wearable_Edge import link_wearable_to_edge
from Server import RxGPUSystem
from Wireless import WiFi

def link_edge_to_server():

    print("\nWhere would you like to perform decompression of the samples?")
    print("1. At the Edge")
    print("2. At the Server")
    while True:
        try:
            decompression_choice = int(input("Enter choice (1 or 2): "))
            if decompression_choice in [1, 2]:
                break
            print("Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    edge_summary = link_wearable_to_edge()
    processed_data = edge_summary.get("Processed Data", [])
    
    print(f"Debug: edge_summary keys: {edge_summary.keys()}")
    print(f"Processed data length in Edge_server.py: {len(processed_data)}")
    if decompression_choice == 2:

        if "Bit Stream" in edge_summary and "Payloads" in edge_summary:
            bit_stream = edge_summary["Bit Stream"]
            payloads = edge_summary["Payloads"]
            print(f"Using raw Bit Stream (length: {len(bit_stream)} bits) and Payloads (count: {len(payloads)}) for server decompression")
            processed_data = []  
            adc_bits = edge_summary.get("ADC Bits", 12)
            num_samples = len(bit_stream) // adc_bits
            data_stream = []
            for i in range(0, len(bit_stream), adc_bits):
                chunk = bit_stream[i:i + adc_bits]
                if len(chunk) == adc_bits:
                    value = int(chunk, 2)  
                    data_stream.append(value)
            processed_data = np.array(data_stream, dtype=np.uint16)
            print(f"Converted Bit Stream to {len(processed_data)} numerical samples for server processing")
        else:
            print("Error: Raw Bit Stream or Payloads not found in edge_summary. Using empty list as fallback.")
            bit_stream = ""
            payloads = []
            processed_data = []
    else:

        if len(processed_data) == 0:
            print("Debug: Checking alternative keys for processed data...")
            if "Reconstructed Samples" in edge_summary:
                recon_samples = edge_summary["Reconstructed Samples"]
                print(f"Debug: 'Reconstructed Samples' value: {recon_samples}, type: {type(recon_samples)}")
                if isinstance(recon_samples, int):
                    print("Warning: 'Reconstructed Samples' is an integer (sample count). Looking for actual data...")
                    if "Reconstructed Data" in edge_summary:
                        processed_data = edge_summary["Reconstructed Data"]
                        print(f"Found 'Reconstructed Data' with length: {len(processed_data)}")
                    else:
                        print("Error: No valid reconstructed data found. Using empty list as fallback.")
                        processed_data = []
                else:
                    processed_data = recon_samples
                    print(f"Found 'Reconstructed Samples' with length: {len(processed_data)}")
            else:
                raise ValueError("Processed data is empty or missing after edge processing. Check Wearable_Edge.py output.")

    bit_stream = edge_summary.get("Bit Stream", "")
    payloads = edge_summary.get("Payloads", [])
    distance = edge_summary.get("Distance", 30)  
    adc_bits = edge_summary.get("ADC Bits", 12)  
    wearable_latency_ms = edge_summary.get("Wearable Latency (ms)", 0.0)
    transmission_to_edge_latency_ms = edge_summary.get("RX Latency (ms)", 0.0)
    edge_decompression_latency_ms = edge_summary.get("Edge Decompression and Processing (ms)", 0.0)
    data_reduction_applied = edge_summary.get("Data Reduction Applied", False)


    reconstructed_samples = []
    bit_stream_length = len(bit_stream)
    edge_rx_power_mw = 0.0
    edge_rx_latency_us = 0.0
    transmission_to_server_latency_ms = 0.0
    server_decompression_latency_ms = 0.0

    if decompression_choice == 1:

        reconstructed_samples = edge_summary.get("Reconstructed Data", [])  
        print("\nDecompression performed at Edge. No further server processing required.")

        start_time = time.perf_counter()
        time.sleep(0.001)  
        end_time = time.perf_counter()
        edge_rx_latency_us = (end_time - start_time) * 1e6
        edge_rx_power_mw = 10.0 

    elif decompression_choice == 2:
        print("\nTransmitting raw data to Server via WiFi for decompression...")

        payload_bytes = len(payloads[0]) if payloads else 1024
        num_payloads = len(payloads) if payloads else 1
        total_bytes = payload_bytes * num_payloads
        mac_overhead_per_payload = 0.1 
        contention_delay = 7  
        protocol_overhead_per_byte = 0.0001  
        overhead_ms = (num_payloads * (mac_overhead_per_payload + contention_delay)) + (total_bytes * protocol_overhead_per_byte)


        base_rate_mbps = 30.0  
        distance_reduction = max(0, (distance - 10) / 10) * 0.1  
        payload_factor = min(1.0, payload_bytes / 1024) 
        default_data_rate_mbps = base_rate_mbps * (1 - distance_reduction) * payload_factor
        print(f"Calculated default_data_rate_mbps: {default_data_rate_mbps:.2f} Mbps (distance: {distance}m, payload: {payload_bytes} bytes)")

        num_samples = len(processed_data) if len(processed_data) > 0 else len(bit_stream) // adc_bits
        total_bits = len(bit_stream) if bit_stream else num_samples * adc_bits * num_payloads
        distance_factor = distance * 0.01  

        wifi_module = WiFi(
            distance=distance,
            data_stream=processed_data, 
            adc_bits=adc_bits,
            payload_bytes=payload_bytes,
            data_rate_mbps=default_data_rate_mbps
        )
        wifi_module.num_payloads = num_payloads


        data_rate_mbps =wifi_module._data_rate_mbps  
        transmission_to_server_latency_ms = (total_bits / (data_rate_mbps * 1e6)) * 1000 + distance_factor + overhead_ms


        start_time = time.perf_counter()
        current_latency_ms = (time.perf_counter() - start_time) * 1000
        if current_latency_ms < transmission_to_server_latency_ms:
            sleep_time = (transmission_to_server_latency_ms - current_latency_ms) / 1000
            time.sleep(sleep_time)
        end_time = time.perf_counter()
        edge_rx_latency_us = (end_time - start_time) * 1e6  
        edge_rx_power_mw = wifi_module.Tx_power_noise(wifi_module.payload_bytes) * 1000  


        print("\nServer processing started...")
        server_rx = RxGPUSystem(wifi_module)
        reconstructed_samples = server_rx.reconstructed_data.tolist()


        server_decompression_latency_ms = (server_rx.rx_latency_us / 1e3) 


    print("\n=== Edge-Server Link Metrics ===")
    print(f"{'Edge RX Latency':<30}: {edge_rx_latency_us:.2f} µs")
    print(f"{'Edge RX Power':<30}: {edge_rx_power_mw:.2f} mW")
    print(f"{'Edge Decompression Latency':<30}: {edge_decompression_latency_ms:.2f} ms")
    print(f"{'Bit Stream Length':<30}: {bit_stream_length} bits")
    print(f"{'Reconstructed Samples':<30}: {len(reconstructed_samples)} samples")
    print("================================\n")


    if decompression_choice == 1:

        labels = [
            "Transmission to Edge",
            "Edge Decompression",
        ]
        latencies = [
            transmission_to_edge_latency_ms,
            edge_decompression_latency_ms,
        ]
        colors = ["#ff7f0e", "#2ca02c"]
    else:

        labels = [
            "Transmission to Edge",
            "Edge Decompression",
            "Transmission to Server",
            "Server Decompression",
        ]
        latencies = [
            transmission_to_edge_latency_ms,
            edge_decompression_latency_ms,
            transmission_to_server_latency_ms,
            server_decompression_latency_ms,
        ]
        colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, latencies, color=colors)
    plt.ylabel("Latency (ms)")
    plt.title("Latency Breakdown of Ultrasound System (Wearable → Edge → Server)")
    plt.xticks(rotation=15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)


    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f"{height:.2f} ms", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


    summary = {
        "Edge RX Latency (µs)": edge_rx_latency_us,
        "Edge RX Power (mW)": edge_rx_power_mw,
        "Edge Decompression Latency (ms)": edge_decompression_latency_ms,
        "Transmission to Server Latency (ms)": transmission_to_server_latency_ms,
        "Server Decompression Latency (ms)": server_decompression_latency_ms,
        "Bit Stream Length": bit_stream_length,
        "Reconstructed Samples": len(reconstructed_samples),
        "Reconstructed Data": reconstructed_samples
    }
    return summary

if __name__ == "__main__":
    summary = link_edge_to_server()
    print("\nEdge-Server Linking Complete. Final Summary:")
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")



