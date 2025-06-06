import numpy as np
import matplotlib.pyplot as plt
from Ultrasound_system import input_calculation
from Sensor_output_analysis import Sensors_outputs_analysis
from Channel_subsampling import greedy_energy_based_channel_subsampling
from mux import mux_sensor_data
from ADC import SAR_ADC
from Temporal import temporal_subsampling_digital
from Compressive_sensing import compressive_sensing_digital
from Rx_beamforming import beamformer_analysis
from MCU import mcu_analysis
from Wireless import Bluetooth, WiFi

def main():
    channel_duration_cache = {}

    # Step 1: Generate data from ultrasound simulation
    ultrasound_data = input_calculation(num_sensors=32, fs_sensors=10e6)

    # Step 2: Analyze the sensor outputs
    num_sensors, signal_params, summary, sensor_angles = Sensors_outputs_analysis(ultrasound_data)
    
    # AFE latency 
    try:
        time_afe_us = summary["sensor_analysis_latency"]
        if time_afe_us < 1e-3:  
            time_afe_us *= 1e6
    except KeyError:
        try:
            time_afe_us = summary["total_processing_us"]
        except KeyError:
            print("Error: Could not find AFE latency in summary. Available keys:", summary.keys())
            raise KeyError("AFE latency key not found in summary dictionary")
    
    power_afe = summary.get("sensor_analysis_power_mw", summary.get("power_estimates", {}).get("total_power_mw", 0.0))

    # Step 3: Subsampling parameters
    num_select = int(num_sensors / 2)  
    fs = summary["fs"]
    probe_width = summary["probe_width"]
    angular_range = summary["angular_range"]
    sensor_angles = np.array(sensor_angles)

    # Step 4: Perform greedy energy-based channel subsampling
    subsampling_results = greedy_energy_based_channel_subsampling(
        signal_params=signal_params,
        num_select=num_select,
        fs=fs,
        sensor_angles=sensor_angles
    )
    subsampled_signal_params = subsampling_results['subsampled_signal_params']
    selected_indices = subsampling_results['selected_indices']

    # Step 5: Compute sensor positions as 1D distances
    radius = probe_width / (2 * np.sin(angular_range / 2))
    sensor_positions_2d = radius * np.column_stack((np.sin(sensor_angles), np.cos(sensor_angles)))
    sensor_positions = sensor_positions_2d[:, 0]
    sensor_positions = sensor_positions[selected_indices]

    # Step 6: Ask user if they want to perform data reduction
    print ("------------------------------------------------------------------------------------------------------------------")
    print ("------------------------------------------------------------------------------------------------------------------\n")
    
    print("\nDo you want to perform data reduction techniques (Temporal Subsampling, Compressive Sensing, or Beamforming)?")
    print("1. Yes")
    print("2. No")
    while True:
        try:
            data_reduction_choice = int(input("Enter choice (1 or 2): "))
            if data_reduction_choice in [1, 2]:
                break
            print("Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # Set data reduction flag
    data_reduction_applied = (data_reduction_choice == 1)

    # Step 7: Set wireless module to Bluetooth (no user input)
    wireless_choice = 1  
    print("\n---------------------------Wireless module set to Bluetooth---------------------------")

    distance = 30  
    
    
    # Step 8: MUX - User selects a channel
    while True:
        try:
            user_choice = int(input(f"\nSelect a channel to pass to ADC (1 to {num_select}): "))
            if 1 <= user_choice <= num_select:
                break
            print(f"Please enter a number between 1 and {num_select}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    selected_channel_index = user_choice - 1
    selected_signal_params, mux_trace = mux_sensor_data(subsampled_signal_params, strategy='static', user_selection=selected_channel_index)

    if user_choice in channel_duration_cache:
        selected_signal_params['duration'] = channel_duration_cache[user_choice]
    else:
        channel_duration_cache[user_choice] = selected_signal_params['duration']

    # Step 9: ADC Processing
    adc_bits = 12 
    adc = SAR_ADC(vref=1.0, vdd=2.5, fs=fs)
    adc.signal_params = selected_signal_params
    t, vin, digital, recon, snr, enob_value, time_adc_us, power_adc = adc.report()

    print(f"\n=================================ADC Results=================================")
    print(f"Power: {power_adc:.2f} mW")
    print(f"ENOB: {enob_value:.2f}")
    print(f"SNR: {snr:.2f} dB")
    print(f"Latency: {time_adc_us:.2f} µs")
    
    print ("------------------------------------------------------------------------------------------------------------------")
    print ("------------------------------------------------------------------------------------------------------------------\n")


    initial_samples = len(digital)
    bit_range = np.arange(6, 15)
    sar_powers = [power_adc * (2.0 ** (bits - 12)) for bits in bit_range]
    sar_enobs = [bits - 0.5 for bits in bit_range]
    sar_snrs = [6.02 * bits + 1.76 for bits in bit_range]
    sar_latencies_us = [time_adc_us * (2.0 ** (bits - 12)) for bits in bit_range]


    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(bit_range, sar_powers, marker='o', label='SAR')
    plt.title("Power vs Resolution")
    plt.xlabel("Bits")
    plt.ylabel("Power (mW)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(bit_range, sar_enobs, marker='o', label='SAR')
    plt.title("ENOB vs Resolution")
    plt.xlabel("Bits")
    plt.ylabel("ENOB")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(bit_range, sar_snrs, marker='o', label='SAR')
    plt.title("SNR vs Resolution")
    plt.xlabel("Bits")
    plt.ylabel("SNR (dB)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(bit_range, sar_latencies_us, marker='o', label='SAR')
    plt.title("Latency vs Resolution")
    plt.xlabel("Bits")
    plt.ylabel("Latency (µs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if data_reduction_choice == 1:  # User wants data reduction
        # Step 10: Choose digital processing technique
        print("\nChoose digital processing technique after channel subsampling:")
        print("1. Temporal Subsampling")
        print("2. Compressive Sensing")
        print("3. Beamforming (Delay and Sum)")
        while True:
            try:
                method_choice = int(input("Enter choice (1, 2 or 3): "))
                if method_choice in [1, 2, 3]:
                    break
                print("Please enter 1, 2 or 3.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        reduction_technique = {1: "Temporal Subsampling", 2: "Compressive Sensing", 3: "Beamforming (Delay and Sum)"}
        selected_technique = reduction_technique.get(method_choice, "None")

        if method_choice in [1, 2]:  # Temporal Subsampling or Compressive Sensing
            if method_choice == 1:
                subsample_factor = 3
                temporal_subsampling_results = temporal_subsampling_digital(
                    adc_digital_data=digital,
                    fs=fs,
                    adc_bits=adc.bits,
                    subsample_factor=subsample_factor
                )
                processed_data = digital[::subsample_factor]

            elif method_choice == 2:
                compression_ratio = 0.50
                compressive_sensing_results = compressive_sensing_digital(
                    adc_digital_data=digital,
                    adc_bits=adc.bits,
                    compression_ratio=compression_ratio
                )
                compressed_signal = compressive_sensing_results["compressed_signal"]
                max_adc_value = 2 ** adc.bits - 1
                processed_data = np.clip(compressed_signal, 0, max_adc_value).astype(int)

            # Step 11: MCU Analysis
            time = len(t) / fs
            mcu_results = mcu_analysis(
                adc_bits=adc.bits,
                adc_sampling_frequency=fs,
                time=time,
                adc_data=processed_data,
                adc_latency=time_adc_us * 1e-6
            )
            time_mcu_us = mcu_results.get("Latency After (µs)", 0.0)
            power_mcu = mcu_results.get("Maximum Power (mW)", 0.0)

            # Step 12: Wireless (Bluetooth only)
            payload_bytes = mcu_results["Payload Bytes"]
            data_rate_mbps = mcu_results["Data Rate (Mbps)"]
            wireless_module = Bluetooth(distance=distance, data_stream=processed_data, adc_bits=adc.bits, payload_bytes=payload_bytes, data_rate_mbps=data_rate_mbps)

            # Step 13: Wireless Transmission
            time_tx = wireless_module.transmission_time(payload_bytes)
            time_tx_us = time_tx * 1e6
            power_tx = wireless_module.Tx_power_noise(payload_bytes) * 1000

            # Calculate Total Latency and Power
            total_time_us = time_afe_us + time_adc_us + time_mcu_us + time_tx_us
            total_time_ms = total_time_us / 1000
            total_power = power_afe + power_adc + power_mcu + power_tx

            # Print Wireless Results
            data_rate = wireless_module.calculate_data_rate(payload_bytes)
            payload_bits = wireless_module.payload_bits(payload_bytes)
            tx_energy_uj = wireless_module.Tx_energy(payload_bytes) * 1e6
            print("\n================================Wireless Results================================")
            print(f"MCU Bits: {len(processed_data) * adc.bits} bits")
            print(f"Data Rate: {data_rate:.2f} Mbps")
            print(f"Transmission Time: {time_tx_us:.2f} µs")
            print(f"Payload Bits: {payload_bits} bits")
            print(f"Power: {power_tx:.2f} mW")
            print(f"Energy: {tx_energy_uj:.2f} µJ")
            print ("------------------------------------------------------------------------------------------------------------------")
            print ("------------------------------------------------------------------------------------------------------------------\n")

            # Plot Combined Graph: Power (AFE, ADC, Wireless) vs Data Rate with Battery Life
            data_rates_mbps = np.linspace(0, min(data_rate, 3.0), 50)
            wireless_powers_mw = np.linspace(0, power_tx, len(data_rates_mbps))
            afe_powers_mw = np.linspace(0, power_afe, len(data_rates_mbps))
            adc_powers_mw = np.linspace(0, power_adc, len(data_rates_mbps))
            desired_operational_time_hours = 6  
            average_power_mw = total_power 
            battery_capacity_mwh = average_power_mw * desired_operational_time_hours  # mWh = mW * hours
            mcu_power_mw = power_mcu
            total_powers_mw = [afe + adc + mcu_power_mw + wireless for afe, adc, wireless in zip(afe_powers_mw, adc_powers_mw, wireless_powers_mw)]
            target_max_power_mw = power_tx  
            
            max_battery_life_hours = battery_capacity_mwh/target_max_power_mw
            k = 3 
            battery_life_hours = [max_battery_life_hours * np.exp(-k * (wireless / power_tx)) for wireless in wireless_powers_mw]

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(data_rates_mbps, afe_powers_mw, label='AFE Power', color='blue', linestyle='--')
            ax1.plot(data_rates_mbps, adc_powers_mw, label='ADC Power', color='orange', linestyle='--')
            ax1.plot(data_rates_mbps, wireless_powers_mw, label='Wireless Power', color='red', linestyle='--')
            ax1.set_xlabel('Data Rate (Mbps)')
            ax1.set_ylabel('Power Consumption (mW)')
            ax1.set_ylim(0, 350)
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend(loc='upper left', fontsize=11)

            ax2 = ax1.twinx()
            ax2.plot(data_rates_mbps, battery_life_hours, label='Battery Life', color='green', linewidth=3)
            ax2.set_ylabel('Battery Life (Hours)')
            ax2.set_ylim(0, 12)
            ax2.set_yticks([0, 3, 6, 9, 12])
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='upper right', fontsize=11)

            plt.title('Power Consumption and Battery Life vs Data Rate')
            plt.tight_layout()
            plt.show()

            
            print("\n=================================Latency Breakdown (µs)=================================\n")
            print(f"Time_AFE: {time_afe_us:.2f}")
            print(f"Time_ADC: {time_adc_us:.2f}")
            print(f"Time_MCU: {time_mcu_us:.2f}")
            print(f"Time_TX: {time_tx_us:.2f}")
            print(f"Total_Time: {total_time_ms:.2f} ms")
            print("\n=================================Power Breakdown (mW)=================================\n")
            print(f"Power_AFE: {power_afe:.2f}")
            print(f"Power_ADC: {power_adc:.2f}")
            print(f"Power_MCU: {power_mcu:.2f}")
            print(f"Power_TX: {power_tx:.2f}")
            print(f"Total_Power: {total_power:.2f}")

            # Summary of Data Reduction
            final_samples = len(processed_data)
            reduction_percentage = ((initial_samples - final_samples) / initial_samples) * 100 if initial_samples > 0 else 0
            print("\n=================================Data Reduction Summary=================================\n")
            print(f"Reduction Technique Used: {selected_technique}")
            print(f"Initial ADC Digital Samples: {initial_samples}")
            print(f"Samples After Reduction: {final_samples}")
            print(f"Data Reduction Achieved: {reduction_percentage:.2f}%")

            # Generate bit stream and payloads
            bit_stream = ''.join(f'{sample:012b}' for sample in processed_data)
            bit_chunks = [bit_stream[i:i+8] for i in range(0, len(bit_stream), 8)]
            byte_array = [int(b, 2) for b in bit_chunks]
            payloads = [byte_array[i:i+payload_bytes] for i in range(0, len(byte_array), payload_bytes)]

            print(f"Returning processed_data with {len(processed_data)} samples")
            return processed_data, wireless_module, bit_stream, payloads, distance, adc_bits, total_time_ms, data_reduction_applied

        elif method_choice == 3:  # Beamforming
            # Step 10: ADC Processing for all channels
            all_digital_signals = []
            all_recon_signals = []
            max_latency_us = 0
            adc_powers = []
            for ch_idx, ch_params in enumerate(subsampled_signal_params):
                adc = SAR_ADC(vref=1.0, vdd=2.5, fs=fs)
                adc.signal_params = ch_params
                if ch_idx + 1 in channel_duration_cache:
                    ch_params['duration'] = channel_duration_cache[ch_idx + 1]
                else:
                    channel_duration_cache[ch_idx + 1] = ch_params['duration']
                t, vin, digital, recon, snr, enob_value, time_adc_ch_us, power_adc_ch = adc.report()
                all_digital_signals.append(digital)
                all_recon_signals.append(recon)
                max_latency_us = max(max_latency_us, time_adc_ch_us)
                adc_powers.append(power_adc_ch)

            time_adc_us = max_latency_us
            power_adc = np.mean(adc_powers)

            print(f"\nADC Results (12-bit Resolution, Average Across {num_select} Channels):")
            print(f"Power: {power_adc:.2f} mW")
            print(f"ENOB: {enob_value:.2f}")
            print(f"SNR: {snr:.2f} dB")
            print(f"Latency: {time_adc_us:.2f} µs")

            # Plot all ADC outputs
            plt.figure(figsize=(15, 5 * num_select))
            for ch_idx in range(num_select):
                plt.subplot(num_select, 1, ch_idx + 1)
                plt.plot(t * 1e6, all_recon_signals[ch_idx] * 1e3, label=f'Channel {ch_idx + 1} (Sensor {selected_indices[ch_idx] + 1})', linestyle='--', marker='o')
                plt.xlabel('Time (µs)')
                plt.ylabel('Voltage (mV)')
                plt.title(f'ADC Output for Channel {ch_idx + 1}')
                plt.grid(True)
                plt.legend()
            plt.tight_layout()
            plt.show()

            # Step 11: Ask user for beamforming angle
            angle = 0.0
            # angle = float(input("Enter beamforming steering angle (in degrees): "))

            # Step 12: Perform beamforming
            digital_array = np.vstack(all_digital_signals)
            beamformer_results = beamformer_analysis(
                elements=num_select,
                adc_bits_per_sample=adc.bits,
                data_samples=len(digital_array[0]),
                sampling_freq_hz=fs,
                vin_volts=adc.vref * 2,
                adc_latency_us=time_adc_us,
                addition_time_ns=1.0,
                adc_codes=digital_array,
                sensor_positions=sensor_positions,
                steering_angle=angle
            )

            # Step 13: MCU Analysis
            time = len(t) / fs
            processed_data = beamformer_results["Digital Codes"]
            
            beamform_subsample_factor = 2 
            processed_data = processed_data[::beamform_subsample_factor]
            mcu_results = mcu_analysis(
                adc_bits=adc.bits,
                adc_sampling_frequency=fs,
                time=time,
                adc_data=processed_data,
                adc_latency=time_adc_us * 1e-6
            )
            time_mcu_us = mcu_results.get("Latency After (µs)", 0.0)
            power_mcu = mcu_results.get("Maximum Power (mW)", 0.0)

            beamformed_signal = beamformer_results["Beamformed Signal"][::beamform_subsample_factor]  
            digital_codes = beamformer_results["Digital Codes"][::beamform_subsample_factor]
            t = np.arange(len(beamformed_signal)) / fs * 1e6

           
            plt.figure(figsize=(10, 8))
            plt.step(t, digital_codes, label="Beamformed Digital Codes (Subsampled)", color='red')
            plt.xlabel("Time (µs)")
            plt.ylabel("Digital Code")
            plt.title("Beamformed Output (Digital Codes, Subsampled)")
            plt.grid(True)
            plt.legend()
            plt.show()

            # Step 14: Wireless (Bluetooth only)
            payload_bytes = mcu_results["Payload Bytes"]
            data_rate_mbps = mcu_results["Data Rate (Mbps)"]
            wireless_module = Bluetooth(distance=distance, data_stream=processed_data, adc_bits=adc.bits, payload_bytes=payload_bytes, data_rate_mbps=data_rate_mbps)

            # Step 15: Wireless Transmission
            time_tx = wireless_module.transmission_time(payload_bytes)
            time_tx_us = time_tx * 1e6
            power_tx = wireless_module.Tx_power_noise(payload_bytes) * 1000

            # Calculate Total Latency and Power
            total_time_us = time_afe_us + time_adc_us + time_mcu_us + time_tx_us
            total_time_ms = total_time_us / 1000
            total_power = power_afe + power_adc + power_mcu + power_tx

            data_rate = wireless_module.calculate_data_rate(payload_bytes)
            payload_bits = wireless_module.payload_bits(payload_bytes)
            tx_energy_uj = wireless_module.Tx_energy(payload_bytes) * 1e6
            print("\n=================================Wireless Results=================================\n")
            print(f"MCU Bits: {len(processed_data) * adc.bits} bits")
            print(f"Data Rate: {data_rate:.2f} Mbps")
            print(f"Transmission Time: {time_tx_us:.2f} µs")
            print(f"Payload Bits: {payload_bits} bits")
            print(f"Power: {power_tx:.2f} mW")
            print(f"Energy: {tx_energy_uj:.2f} µJ")


            data_rates_mbps = np.linspace(0, min(data_rate, 3.0), 50)
            wireless_powers_mw = np.linspace(0, power_tx, len(data_rates_mbps))
            afe_powers_mw = np.linspace(0, power_afe, len(data_rates_mbps))
            adc_powers_mw = np.linspace(0, power_adc, len(data_rates_mbps))
            desired_operational_time_hours = 6
            average_power_mw = total_power  
            battery_capacity_mwh = average_power_mw * desired_operational_time_hours
            mcu_power_mw = power_mcu
            total_powers_mw = [afe + adc + mcu_power_mw + wireless for afe, adc, wireless in zip(afe_powers_mw, adc_powers_mw, wireless_powers_mw)]
            target_max_power_mw = power_tx
            max_battery_life_hours = battery_capacity_mwh / target_max_power_mw
            k = 3  
            battery_life_hours = [max_battery_life_hours * np.exp(-k * (wireless / power_tx)) for wireless in wireless_powers_mw]

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(data_rates_mbps, afe_powers_mw, label='AFE Power', color='blue', linestyle='--')
            ax1.plot(data_rates_mbps, adc_powers_mw, label='ADC Power', color='orange', linestyle='-.')
            ax1.plot(data_rates_mbps, wireless_powers_mw, label='Wireless Power', color='red', linestyle='-')
            ax1.set_xlabel('Data Rate (Mbps)')
            ax1.set_ylabel('Power Consumption (mW)')
            ax1.set_ylim(0, 350)
            ax1.grid(True)
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.plot(data_rates_mbps, battery_life_hours, label='Battery Life', color='green', linestyle=':')
            ax2.set_ylabel('Battery Life (Hours)')
            ax2.set_ylim(0, 12)
            ax2.set_yticks([0, 3, 6, 9, 12])
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='upper right')

            plt.title('Power Consumption and Battery Life vs Data Rate')
            plt.show()

  
            print("\n=================================Latency Breakdown (µs)=================================\n")
            print(f"Time_AFE: {time_afe_us:.2f}")
            print(f"Time_ADC: {time_adc_us:.2f}")
            print(f"Time_MCU: {time_mcu_us:.2f}")
            print(f"Time_TX: {time_tx_us:.2f}")
            print(f"Total_Time: {total_time_ms:.2f} ms")
            print("\n=================================Power Breakdown (mW)=================================\n")
            print(f"Power_AFE: {power_afe:.2f}")
            print(f"Power_ADC: {power_adc:.2f}")
            print(f"Power_MCU: {power_mcu:.2f}")
            print(f"Power_TX: {power_tx:.2f}")
            print(f"Total_Power: {total_power:.2f}")

            final_samples = len(processed_data)
            reduction_percentage = ((initial_samples - final_samples) / initial_samples) * 100 if initial_samples > 0 else 0
            print("\n=================================Data Reduction Summary=================================\n")
            print(f"Reduction Technique Used: {selected_technique}")
            print(f"Initial ADC Digital Samples: {initial_samples}")
            print(f"Samples After Reduction: {final_samples}")
            print(f"Data Reduction Achieved: {reduction_percentage:.2f}%")

            # Generate bit stream and payloads
            bit_stream = ''.join(f'{sample:012b}' for sample in processed_data)
            bit_chunks = [bit_stream[i:i+8] for i in range(0, len(bit_stream), 8)]
            byte_array = [int(b, 2) for b in bit_chunks]
            payloads = [byte_array[i:i+payload_bytes] for i in range(0, len(byte_array), payload_bytes)]

            print(f"Returning processed_data with {len(processed_data)} samples")
            return processed_data, wireless_module, bit_stream, payloads, distance, adc_bits, total_time_ms, data_reduction_applied
        
    else:  # No data reduction
        # Step 10: MCU Analysis
        time = len(t) / fs
        processed_data = digital
        mcu_results = mcu_analysis(
            adc_bits=adc.bits,
            adc_sampling_frequency=fs,
            time=time,
            adc_data=processed_data,
            adc_latency=time_adc_us * 1e-6
        )
        time_mcu_us = mcu_results.get("Latency After (µs)", 0.0)
        power_mcu = mcu_results.get("Maximum Power (mW)", 0.0)

        # Step 11: Wireless (Bluetooth only)
        payload_bytes = mcu_results["Payload Bytes"]
        data_rate_mbps = mcu_results["Data Rate (Mbps)"]
        wireless_module = Bluetooth(distance=distance, data_stream=processed_data, adc_bits=adc.bits, payload_bytes=payload_bytes, data_rate_mbps=data_rate_mbps)

        # Step 12: Wireless Transmission
        time_tx = wireless_module.transmission_time(payload_bytes)
        time_tx_us = time_tx * 1e6
        power_tx = wireless_module.Tx_power_noise(payload_bytes) * 1000

        # Calculate Total Latency and Power
        total_time_us = time_afe_us + time_adc_us + time_mcu_us + time_tx_us
        total_time_ms = total_time_us / 1000
        total_power = power_afe + power_adc + power_mcu + power_tx

        # Print Wireless Results
        data_rate = wireless_module.calculate_data_rate(payload_bytes)
        payload_bits = wireless_module.payload_bits(payload_bytes)
        tx_energy_uj = wireless_module.Tx_energy(payload_bytes) * 1e6
        print("\n=================================Wireless Results=================================")
        print(f"MCU Bits: {len(processed_data) * adc.bits} bits")
        print(f"Data Rate: {data_rate:.2f} Mbps")
        print(f"Transmission Time: {time_tx_us:.2f} µs")
        print(f"Payload Bits: {payload_bits} bits")
        print(f"Power: {power_tx:.2f} mW")
        print(f"Energy: {tx_energy_uj:.2f} µJ")

        # Plot Combined Graph
        data_rates_mbps = np.linspace(0, min(data_rate, 3.0), 50)
        wireless_powers_mw = np.linspace(0, power_tx, len(data_rates_mbps))
        afe_powers_mw = np.linspace(0, power_afe, len(data_rates_mbps))
        adc_powers_mw = np.linspace(0, power_adc, len(data_rates_mbps))
        desired_operational_time_hours = 5  
        average_power_mw = total_power  
        battery_capacity_mwh = average_power_mw * desired_operational_time_hours
        mcu_power_mw = power_mcu
        total_powers_mw = [afe + adc + mcu_power_mw + wireless for afe, adc, wireless in zip(afe_powers_mw, adc_powers_mw, wireless_powers_mw)]
        target_max_power_mw = power_tx
        max_battery_life_hours = battery_capacity_mwh / target_max_power_mw
        k = 3  
        battery_life_hours = [max_battery_life_hours * np.exp(-k * (wireless / power_tx)) for wireless in wireless_powers_mw]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(data_rates_mbps, afe_powers_mw, label='AFE Power', color='blue', linestyle='--')
        ax1.plot(data_rates_mbps, adc_powers_mw, label='ADC Power', color='orange', linestyle='-.')
        ax1.plot(data_rates_mbps, wireless_powers_mw, label='Wireless Power', color='red', linestyle='-')
        ax1.set_xlabel('Data Rate (Mbps)')
        ax1.set_ylabel('Power Consumption (mW)')
        ax1.set_ylim(0, 350)
        ax1.grid(True)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(data_rates_mbps, battery_life_hours, label='Battery Life', color='green', linestyle=':')
        ax2.set_ylabel('Battery Life (Hours)')
        ax2.set_ylim(0, 12)
        ax2.set_yticks([0, 3, 6, 9, 12])
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.legend(loc='upper right')

        plt.title('Power Consumption and Battery Life vs Data Rate')
        plt.show()


        print("\n=================================Latency Breakdown (µs)=================================\n")
        print(f"Time_AFE: {time_afe_us:.2f}")
        print(f"Time_ADC: {time_adc_us:.2f}")
        print(f"Time_MCU: {time_mcu_us:.2f}")
        print(f"Time_TX: {time_tx_us:.2f}")
        print(f"Total_Time: {total_time_ms:.2f} ms")
        print("\n=================================Power Breakdown (mW)=================================\n")
        print(f"Power_AFE: {power_afe:.2f}")
        print(f"Power_ADC: {power_adc:.2f}")
        print(f"Power_MCU: {power_mcu:.2f}")
        print(f"Power_TX: {power_tx:.2f}")
        print(f"Total_Power: {total_power:.2f}")


        final_samples = len(processed_data)
        reduction_percentage = ((initial_samples - final_samples) / initial_samples) * 100 if initial_samples > 0 else 0
        print("\n=================================Data Reduction Summary=================================\n")
        print(f"Reduction Technique Used: None")
        print(f"Initial ADC Digital Samples: {initial_samples}")
        print(f"Samples After Reduction: {final_samples}")
        print(f"Data Reduction Achieved: {reduction_percentage:.2f}%")

        # Generate bit stream and payloads
        bit_stream = ''.join(f'{sample:012b}' for sample in processed_data)
        bit_chunks = [bit_stream[i:i+8] for i in range(0, len(bit_stream), 8)]
        byte_array = [int(b, 2) for b in bit_chunks]
        payloads = [byte_array[i:i+payload_bytes] for i in range(0, len(byte_array), payload_bytes)]

        print(f"Returning processed_data with {len(processed_data)} samples")
        return processed_data, wireless_module, bit_stream, payloads, distance, adc_bits, total_time_ms, data_reduction_applied

if __name__ == "__main__":
    main()