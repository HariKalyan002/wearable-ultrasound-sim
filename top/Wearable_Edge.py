import numpy as np
import matplotlib.pyplot as plt
from Wearable import main as wearable_main
from Edge import process_received_data

def link_wearable_to_edge():

    processed_data, wireless_module, bit_stream, payloads, distance, adc_bits, total_time_ms, data_reduction_applied = wearable_main()

    print(f"Processed data length in Wearable_edge.py: {len(processed_data)}")
    if len(processed_data) == 0:
        raise ValueError("Processed data is empty. Check Wearable.py execution for errors.")

    wireless_choice = 1  

    received_summary = process_received_data(bit_stream, payloads, wireless_choice, distance, adc_bits, data_reduction_applied, decompression_at_server=False)


    transmission_latency_ms = received_summary["RX Latency (ms)"]
    edge_processing_latency_ms = received_summary["Edge Decompression and Processing (ms)"]
    

    labels = ["Wearable Latency", "Transmission Latency", "Edge Decompression & Processing"]
    latencies = [total_time_ms, transmission_latency_ms, edge_processing_latency_ms]
    colors = ['blue', 'Orange', 'Green']

    plt.figure(figsize=(10, 6))
    plt.bar(labels, latencies, color=colors)
    plt.ylabel("Latency (ms)")
    plt.title("Latency Breakdown of Ultrasound System")
    plt.xticks(rotation=15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    

    for i, v in enumerate(latencies):
        plt.text(i, v + 0.5, f"{v:.2f} ms", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

    return received_summary

if __name__ == "__main__":
    link_wearable_to_edge()
    # print("===============================Edge Receiver Summary===============================")
    # for key, value in summary.items():
    #     if isinstance(value, (int, float)):
    #         print(f"{key}: {value:.2f}")
    #     else:
    #         print(f"{key}: {value}")