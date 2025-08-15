
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from ipywidgets import interactive
from ipywidgets import interactive_output, HBox, VBox
import matplotlib.ticker as ticker 



def plot_data_vs_data(x_values, y_values, z_values, length, x_axis_name, y_axis_name='y', y_multiplier=1, log=False):
    x_values_flat = np.array(x_values).flatten()
    y_values_flat = np.array(y_values, dtype=np.float64).flatten()
    z_values_flat = np.array(z_values, dtype=np.float64).flatten()
    
    length_arr = np.array(length)
    
    length_flat = length_arr.flatten()
    

    unique_lengths = np.unique(length_flat)
    unique_lengths_in_micro = unique_lengths * 1e6

    def update_plot(selected_length, x_value=None, y_value=None):
        plt.figure(figsize=(12, 8))  # Make the figure wider (adjust as needed)

        if selected_length == "Show All":
            mask = np.ones_like(length_flat, dtype=bool)
        else:
            selected_length_in_micro = float(selected_length.replace(' μm', ''))
            tolerance = 0.01  # Tighten the tolerance to avoid unwanted data points
            mask = np.abs(length_flat * 1e6 - selected_length_in_micro) < tolerance

        # Apply the mask to the data
        x_values_for_length = x_values_flat[mask]
        y_values_for_length = y_values_flat[mask] * y_multiplier
        z_values_for_length = z_values_flat[mask]
        length_for_length = length_flat[mask] * 1e6

        if selected_length == "Show All":
            for length_value in np.unique(length_for_length):
                mask_all = (length_for_length == length_value)
                plt.plot(x_values_for_length[mask_all], y_values_for_length[mask_all])

            min_length = np.min(unique_lengths_in_micro)
            max_length = np.max(unique_lengths_in_micro)
            plt.title(f'{y_axis_name} vs {x_axis_name} (Length from {min_length:.2f} μm to {max_length:.2f} μm)')

        else:
            plt.plot(x_values_for_length, y_values_for_length)
            plt.title(f'{y_axis_name} vs {x_axis_name} for {selected_length}')

        plt.xlabel(f'{x_axis_name}')
        plt.ylabel(f'{y_axis_name}')

        if log:
            plt.yscale('log')
            plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[], numticks=10))
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'$10^{int(np.log10(x))}$'))
            plt.ylabel(f'{y_axis_name} (Log Base 10)')

        if y_value is not None and x_value_widget.disabled:
            closest_index = np.abs(y_values_for_length - y_value).argmin()
            closest_x = x_values_for_length[closest_index]
            closest_y = y_values_for_length[closest_index]
            corresponding_z = z_values_for_length[closest_index]

            plt.scatter(closest_x, closest_y, color='blue', label=f'Point ({closest_x:.2f}, {closest_y:.2f})')
            z_value_widget.value = corresponding_z
            print(f"The corresponding {x_axis_name} value for {y_axis_name} = {closest_y:.2f} is: {closest_x:.2f}")
        elif x_value is not None and y_value_widget.disabled:
            closest_index = np.abs(x_values_for_length - x_value).argmin()
            closest_x = x_values_for_length[closest_index]
            closest_y = y_values_for_length[closest_index]
            corresponding_z = z_values_for_length[closest_index]

            plt.scatter(closest_x, closest_y, color='red', label=f'Point ({closest_x:.2f}, {closest_y:.2f})')
            z_value_widget.value = corresponding_z
            print(f"The corresponding {y_axis_name} value for {x_axis_name} = {closest_x:.2f} is: {closest_y:.2f}")

        plt.grid(True)
        plt.legend()
        plt.show()

    dropdown_options = ["Show All"] + [f'{length:.2f} μm' for length in unique_lengths_in_micro]
    length_widget = widgets.Dropdown(
        options=dropdown_options,
        value=dropdown_options[0],
        description='Length:',
        layout=widgets.Layout(width='500px')  # Make the dropdown wider
    )

    x_value_widget = widgets.FloatText(
        value=np.mean(x_values_flat),
        description=f"{x_axis_name}:",
        disabled=False,
        layout=widgets.Layout(width='300px', margin='0 40px 0 0'),  # Push input boxes more to the right
        description_width='150px'  # Smaller description width
    )

    y_value_widget = widgets.FloatText(
        value=None,
        description=f"{y_axis_name}:",
        disabled=True,
        layout=widgets.Layout(width='300px', margin='0 40px 0 0'),  # Push input boxes more to the right
        description_width='150px'  # Smaller description width
    )

    z_value_widget = widgets.FloatText(
        value=None,
        description=f" Vgs:",
        disabled=True,
        layout=widgets.Layout(width='300px', margin='0 40px 0 0'),  # Push input boxes more to the right
        description_width='150px'  # Smaller description width
    )

    select_x_or_y_widget = widgets.Checkbox(
        value=True,
        description=f"{x_axis_name} (uncheck for {y_axis_name})",
        layout=widgets.Layout(width='300px')  # Make the checkbox wider
    )

    def toggle_x_or_y(change):
        if change['new']:
            x_value_widget.disabled = False
            y_value_widget.disabled = True
        else:
            x_value_widget.disabled = True
            y_value_widget.disabled = False

    select_x_or_y_widget.observe(toggle_x_or_y, names='value')

    output = interactive_output(update_plot, {
        'selected_length': length_widget,
        'x_value': x_value_widget,
        'y_value': y_value_widget
    })

    display(VBox([length_widget, select_x_or_y_widget, HBox([x_value_widget, y_value_widget]), z_value_widget, output]))



def tile_length_to_match_data(length_array, data_array):
    length_array = np.array(length_array).flatten()  
    data_shape = data_array.shape 
    
    if length_array.size == data_shape[0]:
        # length matches number of rows, repeat along columns
        return np.tile(length_array.reshape(-1, 1), (1, data_shape[1]))
    elif length_array.size == data_shape[1]:
        # length matches number of columns, repeat along rows
        return np.tile(length_array.reshape(1, -1), (data_shape[0], 1))
    else:
        raise ValueError(f"Length array size {length_array.size} does not match any dimension of data shape {data_shape}")

