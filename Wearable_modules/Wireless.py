import numpy as np
import math

class Wireless:
    def __init__(self, frequency_hz, bandwidth_hz, data_rate_mbps, distance, energy_per_bit=55e-10, data_stream=None, adc_bits=12, max_payload_size=200, payload_bytes=None):
        self.frequency_hz = frequency_hz
        self.bandwidth_hz = bandwidth_hz
        self._data_rate_mbps = data_rate_mbps  
        self.distance = distance
        self.speed_of_light = 3e8
        self.energy_per_bit = energy_per_bit
        self.data_stream = data_stream  # MCU output bit stream
        self.adc_bits = adc_bits  # Number of bits per sample from ADC
        self.max_payload_size = max_payload_size
        self.payload_bytes = payload_bytes  
        self.num_payloads = None
        self._dynamic_data_rate_mbps = data_rate_mbps  
        self.process_data_stream()

    def process_data_stream(self):
        """Convert the MCU data stream into payloads for transmission or use provided payload_bytes."""
        if self.payload_bytes is not None:
            # Use the provided payload_bytes (from MCU) and calculate number of payloads
            if self.data_stream is not None:
                total_bits = len(self.data_stream) * self.adc_bits
                total_bytes = total_bits // 8
                if total_bits % 8 != 0:
                    total_bytes += 1
                num_payloads = total_bytes // self.payload_bytes
                if total_bytes % self.payload_bytes != 0:
                    num_payloads += 1
                self.num_payloads = num_payloads
            else:
                self.num_payloads = 1  
            return

        if self.data_stream is None:
            return

        # Convert data stream (digital codes) into bytes
        total_bits = len(self.data_stream) * self.adc_bits
        total_bytes = total_bits // 8
        if total_bits % 8 != 0:
            total_bytes += 1 

        num_payloads = total_bytes // self.max_payload_size
        if total_bytes % self.max_payload_size != 0:
            num_payloads += 1

        self.payload_bytes = min(total_bytes, self.max_payload_size)
        self.num_payloads = num_payloads

    def calculate_data_rate(self, payload_bytes):
        """Calculate dynamic data rate based on SNR and bandwidth using Shannon-Hartley theorem."""
        if payload_bytes is None:
            return self._dynamic_data_rate_mbps
        snr_db = self.snr(payload_bytes)
        snr_linear = 10 ** (snr_db / 10)
        max_data_rate_bps = self.bandwidth_hz * math.log2(1 + snr_linear)  
        data_rate_mbps = max_data_rate_bps / 1e6  
        
        self._dynamic_data_rate_mbps = min(data_rate_mbps, self._data_rate_mbps)
        return self._dynamic_data_rate_mbps

    def bit_time(self, payload_bytes=None):
        """Calculate bit time based on current data rate."""
        if payload_bytes is not None:
           
            pass  
        return 1 / (self._dynamic_data_rate_mbps * 1e6)

    def propagation_time(self):
        return self.distance / self.speed_of_light

    def payload_bits(self, payload_bytes):
        return payload_bytes * 8

    def _base_transmission_time(self, payload_bytes):
        total_bits = self.payload_bits(payload_bytes) + 58 * 8  # Overhead to 58 bytes (Based on Research Paper - IJACSA)
        return total_bits * self.bit_time(payload_bytes) + self.propagation_time()

    def Tx_energy(self, payload_bytes):
        return self.energy_per_bit * self.payload_bits(payload_bytes)

    def power_noise(self):
        k = 1.38e-23  # Boltzmann Constant
        t = 290  # Kelvin
        return k * t * self.bandwidth_hz  # Thermal noise power

    def Tx_power_noise(self, payload_bytes):
        E_tx = self.Tx_energy(payload_bytes)
        T_tx = self._base_transmission_time(payload_bytes)
        return E_tx / T_tx if T_tx > 0 else 0

    def snr(self, payload_bytes):
        P_tx = self.Tx_power_noise(payload_bytes)
        P_noise = self.power_noise()
        if P_noise == 0 or P_tx == 0:
            return float('inf')
        return 10 * math.log10(P_tx / P_noise)  # SNR in dB

    def transmission_time(self, payload_bytes):
        return self._base_transmission_time(payload_bytes)

    def compute_power_profile(self, payload_range):
        tx_powers_mw = []
        transmission_times_us = []
        for payload in payload_range:
            power_mw = self.Tx_power_noise(payload) * 1000  
            tx_time_us = self.transmission_time(payload) * 1e6  
            tx_powers_mw.append(power_mw)
            transmission_times_us.append(tx_time_us)
        return tx_powers_mw, transmission_times_us

    def print_summary(self, payload_bytes=None):
        if payload_bytes is None:
            payload_bytes = self.payload_bytes if self.payload_bytes is not None else 200 
        self.calculate_data_rate(payload_bytes)  

        print("=== Wireless Link Summary ===")
        print(f"{'Frequency':<24}: {self.frequency_hz / 1e9:>8.2f} GHz")
        print(f"{'Bandwidth':<24}: {self.bandwidth_hz / 1e6:>8.2f} MHz")
        print(f"{'Data Rate':<24}: {self._dynamic_data_rate_mbps:>8.2f} Mbps")
        print(f"{'Distance':<24}: {self.distance:>8.2f} meters")
        print(f"{'Payload Size':<24}: {payload_bytes:>8} bytes")
        print(f"{'Bit Time':<24}: {self.bit_time(payload_bytes) * 1e9:>8.2f} ns")
        print(f"{'Transmission Time':<24}: {self.transmission_time(payload_bytes) * 1e6:>8.2f} µs")
        print(f"{'Tx Energy':<24}: {self.Tx_energy(payload_bytes) * 1e6:>8.4f} µJ")
        print(f"{'Tx Power':<24}: {self.Tx_power_noise(payload_bytes) * 1000:>8.2f} mW")
        if self.num_payloads is not None:
            print(f"{'Total Payloads':<24}: {self.num_payloads:>8} packets")
        print("==============================")
        
    def adjust_power_for_data_rate(self, target_data_rate_mbps, payload_bytes):

        base_power_mw = 50  # in mW
        power_mw = base_power_mw + 2 * math.log2(target_data_rate_mbps + 1)
        return power_mw
   
class Bluetooth(Wireless):
    def __init__(self, distance, data_stream=None, adc_bits=12, max_payload_size=200, payload_bytes=None, data_rate_mbps=80):
        super().__init__(
            frequency_hz=2.4e9,           
            bandwidth_hz=2e6,             
            data_rate_mbps=data_rate_mbps,  
            distance=distance,
            energy_per_bit=70e-10,     
            data_stream=data_stream,
            adc_bits=adc_bits,
            max_payload_size=max_payload_size,
            payload_bytes=payload_bytes
        )

class WiFi(Wireless):
    def __init__(self, distance, data_stream=None, adc_bits=12, max_payload_size=200, payload_bytes=None, data_rate_mbps=54):
        super().__init__(
            frequency_hz=5e9,             
            bandwidth_hz=20e6,           
            data_rate_mbps=data_rate_mbps,  
            distance=distance,
            energy_per_bit=70e-10,        
            data_stream=data_stream,
            adc_bits=adc_bits,
            max_payload_size=max_payload_size,
            payload_bytes=payload_bytes
        )