import numpy as np

def tile_length_to_match_data(length_array, data_array):
    length_array = np.array(length_array).flatten()
    data_shape = data_array.shape

    if length_array.size == data_shape[0]:
        return np.tile(length_array.reshape(-1, 1), (1, data_shape[1]))
    elif length_array.size == data_shape[1]:
        return np.tile(length_array.reshape(1, -1), (data_shape[0], 1))
    else:
        raise ValueError(f"Length array size {length_array.size} does not match any dimension of data shape {data_shape}")

def display_resistance(ro_value):
    if ro_value < 1e3:
        return ro_value, "Ω"
    elif ro_value < 1e6:
        return ro_value / 1e3, "kΩ"
    elif ro_value < 1e9:
        return ro_value / 1e6, "MΩ"
    else:
        return ro_value / 1e9, "GΩ"

def display_current(Id_value):
    if Id_value < 1e-6:
        return Id_value * 1e9, "nA"
    elif Id_value < 1e-3:
        return Id_value * 1e6, "μA"
    else:
        return Id_value * 1e3, "mA"

def dB_to_linear(av_db):
    return 10 ** (av_db / 20)

def determine_inversion_region(gm_id_value, device_type):
    if device_type not in ['nmos', 'pmos']:
        raise ValueError("Invalid device type. Use 'nmos' or 'pmos'.")
    if gm_id_value > 20:
        return "Weak Inversion"
    elif 10 < gm_id_value <= 20:
        return "Moderate Inversion"
    else:
        return "Strong Inversion"