import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal_module

k = 1.38e-23  # Boltzmann constant
T = 300       # Temperature in Kelvin

class BaseADC:
    def __init__(self, bits=12, vref=1.0, vdd=2.5, fs=10e6):
        self.bits = bits
        self.vref = vref
        self.vdd = vdd
        self.fs = fs
        self.lsb = vref / (2 ** bits)
        self.signal = None
        self.signal_params = None

    def compute_snr_db(self, c_dac, signal):
        signal_power = np.mean(signal ** 2)
        thermal_noise_power = 2 * k * T / c_dac
        quantization_noise_power = (self.lsb ** 2) / 12
        total_noise_power = thermal_noise_power + quantization_noise_power
        snr_linear = signal_power / total_noise_power
        return 10 * np.log10(snr_linear) if snr_linear > 0 else 0

    def enob(self, c_dac, signal):
        snr_db = self.compute_snr_db(c_dac, signal)
        return (snr_db - 1.76) / 6.02

    def generate_signal(self):
        duration = 9e-6
        num_samples = int(self.fs * duration)
        t = np.linspace(0, duration, num_samples)
        freq = 1e6
        amplitude = 0.45 * self.vref
        offset = 0.5 * self.vref
        signal = amplitude * np.sin(2 * np.pi * freq * t) + offset
        return t, signal

class SAR_ADC(BaseADC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c_u_min = 1e-15
        self.c_c_min = 10e-15
        self.activity_factor = 0.5
        self.c_gate = 200e-15
        self.n_b = 2000
        self.signal = None
        self.signal_params = None

        temp_c_u = 10e-15
        temp_c_total = temp_c_u * (2 ** self.bits)
        dummy_signal = np.ones(100) * (self.vref / 2)
        snr_db = self.compute_snr_db(temp_c_total, dummy_signal)
        snr_linear = 10 ** (snr_db / 10)

        num = (24 * k * T * snr_db)
        deni = ((2 ** self.bits) * (self.vref ** 2))
        self.c_u = -((num) / (deni)) + self.c_u_min
        self.c_total = (2 ** self.bits) * self.c_u

    def evaluate_signal(self, t):
        raw_signal = np.zeros_like(t)
        for echo in self.signal_params['echoes']:
            strength = echo['strength']
            t_delay = echo['t_delay']
            duration = echo['duration']
            f_ultrasound = echo['f_ultrasound']
            decay = echo['decay']
            envelope = np.exp(-(t - t_delay) / decay) * (t >= t_delay) * (t - t_delay < duration)
            raw_signal += strength * np.sin(2 * np.pi * f_ultrasound * (t - t_delay)) * envelope
        return raw_signal

    def sample_and_hold(self, raw_signal):
        return raw_signal

    def comparator(self, vin, vcmp):
        return vin > vcmp

    def dac(self, code):
        return code * self.lsb

    def sar_logic(self, vin):
        code = 0
        for i in reversed(range(self.bits)):
            trial_code = code | (1 << i)
            vcmp = self.dac(trial_code)
            if self.comparator(vin, vcmp):
                code = trial_code
        return code

    def quantize(self, raw_signal):
        signal_shifted = raw_signal + (self.vref / 2)
        signal_shifted = np.clip(signal_shifted, 0, self.vref)
        return np.array([self.sar_logic(v) for v in signal_shifted])

    def compute_energies(self, signal, num_samples):
        snr_db = self.compute_snr_db(self.c_total, signal)
        snr_linear = 10 ** (snr_db / 10)

        e_dac = 0
        for i in range(1, self.bits):
            term = (2 ** (self.bits - 3 - 2 * i)) * ((2 ** i) - 1) * self.c_u * (self.vref ** 2)
            e_dac += term

        c_c = (24 * k * T * snr_linear) / (self.vref ** 2) + self.c_c_min
        e_comp = (c_c * (self.vdd ** 2)) * self.bits

        e_g = self.activity_factor * self.c_gate * (self.vdd ** 2)
        e_logic = self.n_b * e_g * self.bits

        if signal is not None and len(signal) > 0:
            delta_v = np.abs(np.diff(signal, prepend=signal[0]))
            activity_factor = delta_v / self.vref
            activity_factor = np.clip(activity_factor, 0, 1).mean()
        else:
            activity_factor = self.activity_factor

        e_dac_dynamic = e_dac * activity_factor
        total_energy = (e_dac_dynamic + e_comp + e_logic) * num_samples  
        return total_energy

    def power(self, signal, num_samples):
        total_energy = self.compute_energies(signal, num_samples)
        processing_time = num_samples / self.fs
        power_watts = total_energy / processing_time if processing_time > 0 else 0
        return power_watts * 1e3  

    def latency(self, num_samples):
        print(f"Debug: SAR_ADC latency calculation - bits: {self.bits}, fs: {self.fs}, num_samples: {num_samples}")
        conversion_time_per_sample = self.bits / self.fs  
        # Total latency 
        total_latency_seconds = conversion_time_per_sample * num_samples
        total_latency_us = total_latency_seconds * 1e6
        print(f"Debug: conversion_time_per_sample: {conversion_time_per_sample}, total_latency_us: {total_latency_us}")
        return total_latency_us

    def report(self):
        if self.signal_params is None:
            t, raw_signal = self.generate_signal()
        else:
            duration = self.signal_params['duration']
            num_samples = int(self.fs * duration)
            t = np.linspace(0, duration, num_samples)
            raw_signal = self.evaluate_signal(t)
            if 'filter_coeffs' in self.signal_params:
                b, a = self.signal_params['filter_coeffs']
                raw_signal = signal_module.filtfilt(b, a, raw_signal)

        self.signal = raw_signal
        digital = self.quantize(raw_signal)
        analog_recon = (digital * self.lsb) - (self.vref / 2)
        self.adc_output = analog_recon
        snr = self.compute_snr_db(self.c_total, raw_signal)
        latency_us = self.latency(len(raw_signal))
        power = self.power(raw_signal, len(raw_signal))
        enob_value = self.enob(self.c_total, raw_signal)
        return t, raw_signal, digital, analog_recon, snr, enob_value, latency_us, power

class Flash_ADC(BaseADC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_comparators = 2 ** self.bits - 1
        v_in_pp = self.vref
        signal_power = (v_in_pp ** 2) / 2
        ideal_snr_db = 6.02 * self.bits + 1.76
        self.c_comp = (2 ** self.bits - 1) * (k * T / signal_power) * (10 ** (ideal_snr_db / 10))
        
        self.c_dac = self.c_comp
        self.c_per_comp = 300e-15
        self.vdd = kwargs.get("vdd", 1.0)
        self.fs = kwargs.get("fs", 1e6)
        self.activity_factor = 0.5
        self.c_gate = 10e-15
        self.n_logic_gates = 400

    def resistor_ladder(self):
        return np.linspace(0, self.vref, self.num_comparators + 1)

    def comparator_array(self, vin, thresholds):
        return vin > thresholds

    def priority_encoder(self, comp_outputs):
        return np.argmax(comp_outputs[::-1])

    def quantize(self, signal):
        signal = np.clip(signal, 0, self.vref)
        thresholds = self.resistor_ladder()
        return np.array([self.priority_encoder(self.comparator_array(v, thresholds)) for v in signal])

    def power(self, signal):
        signal = np.clip(signal, 0, self.vref)
        delta_v = np.abs(np.diff(signal, prepend=signal[0]))
        activity_scale = delta_v / self.vref
        comparator_energy = 0.5 * (self.c_per_comp) * (self.vdd ** 2)
        total_energy_comparator = self.num_comparators * comparator_energy * activity_scale
        energy_logic = self.n_logic_gates * self.activity_factor * self.c_gate * (self.vdd**2)
        energy_total = total_energy_comparator + energy_logic
        power_mW = (self.fs * energy_total) * 1000
        return power_mW
    
    def average_power(self, signal):
        power_array = self.power(signal)
        return np.mean(power_array)

    def latency(self):
        print(f"Debug: Flash_ADC latency calculation - fs: {self.fs}")
        t_comparator = 1 / self.fs
        t_encoder = 1 / (10 * self.fs)
        total_latency_us = (t_comparator + t_encoder) * 1e6
        print(f"Debug: total_latency_us: {total_latency_us}")
        return total_latency_us

    def compute_snr_db_flash(self):
        v_in = self.vref
        signal_power = (v_in ** 2) / 2
        noise_power = (2 ** self.bits - 1) * (k * T / self.c_dac)
        snr_linear = signal_power / noise_power
        return 10 * np.log10(snr_linear)

    def enob(self):
        snr_db = self.compute_snr_db_flash()
        return (snr_db - 1.76) / 6.02

    def report(self):
        t, signal = self.generate_signal()
        digital = self.quantize(signal)
        analog_recon = digital * self.lsb
        self.adc_output = analog_recon
        snr = self.compute_snr_db_flash()
        latency_us = self.latency()
        power = self.average_power(signal)
        enob_value = self.enob()
        return t, signal, digital, analog_recon, snr, enob_value, latency_us, power

def analyze_resolution(vref=1.0, vdd=3.3, fs=40e6, bit_range=range(4, 13)):
    sar_powers, sar_enobs, sar_snrs, sar_latencies = [], [], [], []
    flash_powers, flash_enobs, flash_snrs, flash_latencies = [], [], [], []

    for bits in bit_range:
        print(f"\nResolution: {bits} bits")
        
        # SAR ADC
        sar_adc = SAR_ADC(bits=bits, vref=vref, vdd=vdd, fs=fs)
        t, _, digital, _, sar_snr, sar_enob, sar_latency_us, sar_power = sar_adc.report()
        print("SAR ADC Digital Output (first 5 codes in binary):")
        for code in digital[:5]:
            print(f"{code:0{bits}b}")
        sar_powers.append(sar_power)
        sar_enobs.append(sar_enob)
        sar_snrs.append(sar_snr)
        sar_latencies.append(sar_latency_us)

        # Flash ADC
        flash_adc = Flash_ADC(bits=bits, vref=vref, vdd=vdd, fs=fs)
        t, _, digital, _, flash_snr, flash_enob, flash_latency_us, flash_power = flash_adc.report()
        print("Flash ADC Digital Output (first 5 codes in binary):")
        for code in digital[:5]:
            print(f"{code:0{bits}b}")
        flash_powers.append(flash_power)
        flash_enobs.append(flash_enob)
        flash_snrs.append(flash_snr)
        flash_latencies.append(flash_latency_us)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(list(bit_range), sar_powers, marker='o', label='SAR')
    # plt.plot(list(bit_range), flash_powers, marker='o', label='Flash')
    plt.title("Power vs Resolution")
    plt.xlabel("Bits")
    plt.ylabel("Power (mW)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(list(bit_range), sar_enobs, marker='o', label='SAR')
    # plt.plot(list(bit_range), flash_enobs, marker='o', label='Flash')
    plt.title("ENOB vs Resolution")
    plt.xlabel("Bits")
    plt.ylabel("ENOB")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(list(bit_range), sar_snrs, marker='o', label='SAR')
    # plt.plot(list(bit_range), flash_snrs, marker='o', label='Flash')
    plt.title("SNR vs Resolution")
    plt.xlabel("Bits")
    plt.ylabel("SNR (dB)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(list(bit_range), sar_latencies, marker='o', label='SAR')
    # plt.plot(list(bit_range), flash_latencies, marker='o', label='Flash')
    plt.title("Latency vs Resolution")
    plt.xlabel("Bits")
    plt.ylabel("Latency (µs)")
    plt.legend()
    plt.grid(True)

    plt.suptitle("ADC Performance Comparison")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    analyze_resolution()