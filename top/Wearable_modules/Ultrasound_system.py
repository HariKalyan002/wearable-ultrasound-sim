import numpy as np

def input_calculation(
    num_sensors=32,
    fs_sensors=10e6,  
    f_ultrasound_sensor=5e6,
    t_duration_sensor=9e-6,
    transducer_sensitivity_sensor=0.1,
    lna_gain_db_sensor=20,
    cutoff_freq_sensor=4e6, 
    probe_width_sensor=0.01,
    angular_range_sensor=np.deg2rad(60)
):

    c_sensor = 1540
    pulse_duration_sensors = 2e-6
    R_load_sensor = 50
    max_depth_sensor = 0.05  # 5 cm


    reflectors = [
        {'depth': 0.01, 'lateral': 0e-3, 'strength': 0.1},  # 1 cm
        {'depth': 0.03, 'lateral': 2e-3, 'strength': 0.08},  # 3 cm
        {'depth': 0.05, 'lateral': -1e-3, 'strength': 0.05}  # 5 cm
    ]
    reflectors_array = np.array(
        [(r['depth'], r['lateral'], r['strength']) for r in reflectors],
        dtype=[('depth', float), ('lateral', float), ('strength', float)]
    )

    sensor_positions = np.linspace(-probe_width_sensor / 2, probe_width_sensor / 2, num_sensors)
    sensor_angles = np.linspace(-angular_range_sensor / 2, angular_range_sensor / 2, num_sensors)


    sensor_signal_params = []
    for i in range(num_sensors):
        x_sensor = sensor_positions[i]
        theta = sensor_angles[i]
        echoes = []
        max_duration = 0
        for reflector in reflectors:
            depth = reflector['depth']
            lateral = reflector['lateral']
            strength = reflector['strength']
            dx = lateral - x_sensor
            distance = np.sqrt(dx**2 + depth**2)
            t_delay = 2 * distance / c_sensor
            angle_to_reflector = np.arctan2(dx, depth)
            angular_gain = np.cos(angle_to_reflector - theta)
            if angular_gain < 0:
                angular_gain = 0
            
            echo = {
                'strength': strength * angular_gain,
                't_delay': t_delay,
                'duration': pulse_duration_sensors,
                'f_ultrasound': f_ultrasound_sensor,
                'decay': 1e-6  
            }
            echoes.append(echo)

            echo_end = t_delay + pulse_duration_sensors
            if echo_end > max_duration:
                max_duration = echo_end
  
        duration_variation = np.random.uniform(0.8, 1.2)  
        sensor_duration = max_duration * duration_variation
        sensor_signal_params.append({
            'echoes': echoes,
            'duration': sensor_duration
        })

    return {
        "sensor_signal_params": sensor_signal_params,
        "sensor_angles": sensor_angles,
        "num_sensors": num_sensors,
        "fs_sensors": fs_sensors,
        "lna_gain_db_sensor": lna_gain_db_sensor,
        "cutoff_freq_sensor": cutoff_freq_sensor,
        "R_load_sensor": R_load_sensor,
        "reflectors": reflectors,
        "probe_width_sensor": probe_width_sensor,
        "f_ultrasound_sensor": f_ultrasound_sensor,
        "transducer_sensitivity_sensor": transducer_sensitivity_sensor,
        "angular_range_sensor": angular_range_sensor,
        "reflectors_array": reflectors_array,
        "c_sensor": c_sensor
    }