import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import Counter
import math

def mcu_analysis(adc_bits=16, adc_sampling_frequency=50e6, time=0.000001, mcu_freq=48e6, mcu_voltage=3,
                 mcu_current=16.7e-3, adc_data=None, adc_latency=None):
    energy_per_cycle = (mcu_voltage * mcu_current) / mcu_freq

    def total_num_samples(adc_sampling_frequency, time):
        return int(adc_sampling_frequency * time)

    num_samples = total_num_samples(adc_sampling_frequency, time)
    if adc_data is None:
        adc_data = np.random.randint(0, 2**adc_bits, size=num_samples)
    elif len(adc_data) != num_samples:
        print(f"Warning: adc_data length ({len(adc_data)}) does not match expected samples ({num_samples}). Using adc_data length.")
        num_samples = len(adc_data)

    original_bits = num_samples * adc_bits
#     # Huffman Compression

#     class Node:
#         def __init__(self, freq, symbol, left=None, right=None):
#             self.freq = freq
#             self.symbol = symbol
#             self.left = left
#             self.right = right
#             self.huff = ''

#         def __lt__(self, nxt):
#             return self.freq < nxt.freq

#     def build_huffman_tree(data):
#         freq = Counter(data)
#         u = len(freq)
#         print(f"Number of unique symbols: {u}")
#         print(f"Unique values in adc_data: {sorted(np.unique(data))}")
#         print(f"Min/Max adc_data: {int(np.min(data))}/{int(np.max(data))}")
#         heap = [Node(freq[sym], sym) for sym in freq]
#         heapq.heapify(heap)
#         while len(heap) > 1:
#             left = heapq.heappop(heap)
#             right = heapq.heappop(heap)
#             merged = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
#             heapq.heappush(heap, merged)
#         return heap[0] if heap else None

#     def generate_huffman_codes(node, val='', codes=None):
#         if codes is None:
#             codes = {}
#         newVal = val + str(node.huff)
#         if node.left:
#             node.left.huff = '0'
#             generate_huffman_codes(node.left, newVal, codes)
#         if node.right:
#             node.right.huff = '1'
#             generate_huffman_codes(node.right, newVal, codes)
#         if not node.left and not node.right:
#             codes[node.symbol] = newVal or '0'
#         return codes

#     huffman_root = build_huffman_tree(adc_data)
#     if huffman_root is None:
#         compressed_bits = num_samples
#         print("Single symbol detected, no compression applied.")
#     else:
#         huffman_codes = generate_huffman_codes(huffman_root)
#         compressed_bits = sum(len(huffman_codes[sample]) for sample in adc_data)
#         print(f"Huffman code lengths: {[len(code) for code in huffman_codes.values()]}")
# """
    compressed_bits = original_bits
    def payload_bytes_mcu():
        return math.ceil(compressed_bits / 8)

    u = len(set(adc_data))
    # cycles_huffman = u * math.log2(u) if u > 1 else 0
    # cycles_encoding = num_samples
    # cycles_before = cycles_huffman + cycles_encoding

    # Using only encoding cycles 
    cycles_before = num_samples  

    energy_MCU = cycles_before * energy_per_cycle
    power_MCU = energy_MCU / (cycles_before / mcu_freq) if cycles_before > 0 else 0
    latency_before = cycles_before / mcu_freq
    latency_after = compressed_bits / mcu_freq
    
    # Data Rate Calculation
    if latency_after > 0:
        data_rate_bps = compressed_bits / latency_after 
        data_rate_mbps = data_rate_bps / 1e6  
    else:
        data_rate_mbps = 0  


    #processing energy and voltage
    energy_processing = power_MCU * latency_after
    voltage_processing = mcu_voltage

    # --- Maximum Power & Latency Calculations ---
    max_cycles = num_samples  
    max_latency = max_cycles / mcu_freq
    max_power = mcu_voltage * mcu_current  


    print("===========================MCU_results===========================")
    print(f"Original_bits                                  : {original_bits} bits")
    print(f"Compressed_bits                                : {compressed_bits} bits")
    print(f"Power_before                                   : {power_MCU * 1e3:.2f} mW")
    print(f"Latency_after                                  : {latency_after * 1e6:.2f} µs")
    print(f"Payload Bytes (after compression)              : {payload_bytes_mcu()} bytes")
    print(f"Data Rate                                      : {data_rate_mbps:.2f} Mbps")
    print(f"Processing Energy                              : {energy_processing * 1e6:.2f} µJ")
    print(f"Processing Voltage                             : {voltage_processing:.2f} V")
    print(f"Maximum Latency                                : {max_latency * 1e6:.2f} µs")
    print(f"Maximum Power                                  : {max_power * 1e3:.2f} mW")
    print("-----------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------")

    # plt.figure(figsize=(10, 8))
    # x = np.arange(2)
    # width = 0.4
    # plt.subplot(2, 1, 1)
    # plt.bar(x, [original_bits, compressed_bits], width, color=['blue', 'green'])
    # plt.ylabel("Bits")
    # plt.title("Number of Bits")
    # plt.xticks(x, ['Before Huffman', 'After Huffman'])
    # plt.subplot(2, 1, 2)
    # plt.bar(x, [latency_before * 1e6, latency_after * 1e6], width, color=['blue', 'green'])
    # plt.ylabel("Latency (µs)")
    # plt.title("MCU Latency")
    # plt.tight_layout()
    # plt.savefig('mcu_metrics.png')
    # plt.close()

    def total_cycles(adc_data, adc_sampling_frequency, mcu_freq, adc_latency):
            n = len(adc_data)
            execution_time = n / mcu_freq
            return execution_time * mcu_freq

    cycles = total_cycles(adc_data, adc_sampling_frequency, mcu_freq, adc_latency)
    execution_time = cycles / mcu_freq
    print(f"Number of Cycles         : {cycles:.2f}")
    print(f"Execution_time of MCU    : {execution_time * 1e6:.2f} µs")

    return {
        "ADC_bits": adc_bits,
        "ADC Sampling Frequency": adc_sampling_frequency,
        "Time": time,
        "MCU Frequency": mcu_freq,
        "MCU Voltage": mcu_voltage,
        "MCU Current": mcu_current,
        "Original Bits": original_bits,
        "Compressed Bits": compressed_bits,
        "Payload Bytes": payload_bytes_mcu(),
        "Latency Before (µs)": latency_before * 1e6,
        "Latency After (µs)": latency_after * 1e6,
        "Number of Cycles": cycles,
        "Execution Time (s)": execution_time,
        "Processing Energy (J)": energy_processing,
        "Processing Voltage (V)": voltage_processing,
        "Maximum Latency (µs)": max_latency * 1e6,
        "Maximum Power (mW)": max_power * 1e3,
        "Data Rate (Mbps)": data_rate_mbps 
    }