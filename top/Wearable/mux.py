import numpy as np

def mux_sensor_data(signal_params, strategy='static', user_selection=0):
    selected_params = signal_params[user_selection]
    return selected_params, None  
