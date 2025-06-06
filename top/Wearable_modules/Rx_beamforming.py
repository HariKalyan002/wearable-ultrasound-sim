import numpy as np
import matplotlib.pyplot as plt

def serialize_adc_bitstreams(adc_codes, adc_bits):
    adc_codes = np.asarray(adc_codes, dtype=np.int32)  
    elements, samples = adc_codes.shape
    serialized = {}

    for i in range(elements):
        codes = adc_codes[i]
        codes = np.asarray(codes, dtype=np.int32) 
        bit_matrix = ((codes[:, None] & (1 << np.arange(adc_bits)[::-1])) > 0).astype(int)
        bitstream = bit_matrix.flatten()
        serialized[f"Element_{i}"] = bitstream

    return serialized

def deserialize_bitstreams(bitstreams, adc_bits, data_samples):
    """
    Convert serialized bitstreams back to digital codes.
    """
    reconstructed_codes = []
    for bits in bitstreams:
        bit_matrix = np.reshape(bits, (data_samples, adc_bits))
        codes = np.dot(bit_matrix, 1 << np.arange(adc_bits - 1, -1, -1))
        reconstructed_codes.append(codes)
    return np.array(reconstructed_codes)


def beamformer_analysis(elements=None, 
                        adc_bits_per_sample=10, 
                        data_samples=None, 
                        sampling_freq_hz=None, 
                        vin_volts=5.0,
                        adc_latency_us=None,
                        addition_time_ns=1.0,
                        adc_codes=None,
                        sensor_positions=None,
                        steering_angle=0.0,
                        **kwargs):

    adc_codes = np.asarray(adc_codes)

    if adc_codes is None or len(adc_codes) != elements or len(adc_codes[0]) != data_samples:
        raise ValueError(f"adc_codes must be a list/array of {elements} signals, each with {data_samples} samples.")

    # Serializing bitstreams for each element 
    serialized_streams = serialize_adc_bitstreams(adc_codes, adc_bits_per_sample)

    # Deserializing before beamforming 
    adc_codes = deserialize_bitstreams([serialized_streams[f"Element_{i}"] for i in range(elements)], adc_bits_per_sample, data_samples)


    steering_angle_rad = np.deg2rad(steering_angle)

    if sensor_positions is None:
        sensor_spacing = 0.0005
        sensor_positions = np.arange(elements) * sensor_spacing

    growth_bits = int(np.ceil(np.log2(elements))) if elements > 1 else 0
    bf_output_bitwidth = adc_bits_per_sample + growth_bits

    speed_of_sound = 1540
    sample_period = 1 / sampling_freq_hz
    delays = sensor_positions * np.sin(steering_angle_rad) / speed_of_sound
    sample_delays = delays / sample_period
    max_delay = int(np.ceil(np.max(np.abs(sample_delays))))

    aligned_signals = np.zeros((elements, data_samples + max_delay), dtype=np.int32)
    for i in range(elements):
        delay_samples = int(np.round(sample_delays[i]))
        start_idx = max_delay - delay_samples
        aligned_signals[i, start_idx:start_idx + data_samples] = adc_codes[i]

    def full_adder_bitwise(a_bit, b_bit, carry_in):
        sum_bit = a_bit ^ b_bit ^ carry_in
        carry_out = (a_bit & b_bit) | (a_bit & carry_in) | (b_bit & carry_in)
        return sum_bit, carry_out

    def serial_adder_nbit(inputs, n_bits):
        num_inputs, N = inputs.shape
        result = np.zeros(N, dtype=int)
        for t in range(N):
            acc = np.zeros(n_bits + num_inputs, dtype=int)
            carry = 0
            for i in range(num_inputs):
                a = np.binary_repr(inputs[i, t], width=n_bits)
                a_bits = np.array([int(bit) for bit in a[::-1]])
                if i == 0:
                    acc[:n_bits] = a_bits
                else:
                    temp = np.zeros_like(acc)
                    for b in range(n_bits):
                        sum_bit, carry = full_adder_bitwise(acc[b], a_bits[b], carry)
                        temp[b] = sum_bit
                    temp[n_bits] = carry
                    acc = temp.copy()
            result[t] = np.dot(acc, 2 ** np.arange(acc.size))
        return result

    beamformed_codes = serial_adder_nbit(aligned_signals[:, max_delay:max_delay + data_samples], adc_bits_per_sample)

    levels = 2 ** bf_output_bitwidth - 1
    digital_codes = np.clip(beamformed_codes, 0, levels).astype(int)

    offset = vin_volts / 2
    beamformed_signal = digital_codes / levels * vin_volts - offset

    bit_positions = np.arange(bf_output_bitwidth - 1, -1, -1, dtype=np.int32)
    bit_matrix = ((digital_codes[:, None] & np.left_shift(1, bit_positions)) > 0).astype(int)
    bitstream = bit_matrix.flatten()
    
    # delay_time_us = max_delay * sample_period * 1e6
    adds_per_sample = elements - 1
    total_adds = adds_per_sample * data_samples
    addition_time_s = total_adds * (addition_time_ns * 1e-9)
    addition_time_us = addition_time_s * 1e6
    bf_processing_time_us = max_delay / sampling_freq_hz * 1e6
    # latency_us = bf_processing_time_us + addition_time_us

    t = np.arange(data_samples) / sampling_freq_hz * 1e6

    plt.figure(figsize=(10, 4))
    plt.plot(t, beamformed_signal, marker='o', linestyle='-', color='b')
    plt.xlabel('Time (µs)')
    plt.ylabel('Amplitude (V)')
    plt.title('Beamformed Output Signal (Voltage, for visualization)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("=== BEAMFORMER ANALYSIS ===")
    print(f"Number of Elements (ADCs)                            : {elements}")
    print(f"Adds per Output Sample                               : {adds_per_sample}")
    print(f"Total Digital Bits                                   : {data_samples * bf_output_bitwidth:,} bits")
    print(f"Sampling Frequency                                   : {sampling_freq_hz / 1e6:.2f} MHz")
    print("---------------------------------------------------------------------------------------------------")

    return {
        "Elements": elements,
        "Output Bitwidth": bf_output_bitwidth,
        "Total Digital Bits": data_samples * bf_output_bitwidth,
        "Beamformed Signal": beamformed_signal,
        "Digital Codes": digital_codes,
        "BitStream": bitstream,
        "Serialized Inputs": serialized_streams
    }
