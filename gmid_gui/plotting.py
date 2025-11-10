import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from ipywidgets import interactive_output, VBox, HBox
import matplotlib.ticker as ticker
from IPython.display import display


def format_with_si(value):
    """
    Format number with proper SI prefix (f, p, n, µ, m, k, M, G, T, P).
    Works individually per value.
    """
    if value == 0:
        return "0"
    
    prefixes = {
        -15: "f", -12: "p", -9: "n", -6: "µ",
        -3: "m", 0: "", 3: "k", 6: "M",
        9: "G", 12: "T", 15: "P"
    }
    
    exp = int(np.floor(np.log10(abs(value)) / 3) * 3)  # nearest multiple of 3
    exp = max(min(exp, 15), -15)  # clamp to available prefixes
    
    scaled = value / (10 ** exp)
    return f"{scaled:.3g}{prefixes[exp]}"

def format_for_box(value, sig=3):
    """
    Format number for X/Y/Z input boxes:
    - Rounds to given significant digits
    - Uses scientific notation for very small/large values
    """
    if value == 0:
        return 0.0
    exp = int(np.floor(np.log10(abs(value))))
    # if value is very small (<1e-3) or very large (>1e4), use sci notation
    if exp < -3 or exp > 4:
        return float(f"{value:.{sig}e}")
    else:
        return float(f"{value:.{sig}g}")
    



def plot_data_vs_data(x_values, y_values, z_values, length, x_axis_name, y_axis_name='y', y_multiplier=1, log=False):
    x_values_flat = np.array(x_values).flatten()
    y_values_flat = np.array(y_values, dtype=np.float64).flatten()
    z_values_flat = np.array(z_values, dtype=np.float64).flatten()
    length_flat = np.array(length).flatten()
    
    unique_lengths = np.unique(length_flat)
    unique_lengths_in_micro = unique_lengths * 1e6

    def update_plot(selected_length, active_var, x_val, y_val, z_val):
        fig, ax = plt.subplots(figsize=(12, 8))  # create figure
        
        # Length filter
        if selected_length == "Show All":
            mask = np.ones_like(length_flat, dtype=bool)
        else:
            selected_length_in_micro = float(selected_length.replace(' μm', ''))
            mask = np.abs(length_flat * 1e6 - selected_length_in_micro) < 0.01

        x_vals = x_values_flat[mask]
        y_vals = y_values_flat[mask] * y_multiplier
        z_vals = z_values_flat[mask]

        # Plot data
        if selected_length == "Show All":
            for l in np.unique(length_flat[mask] * 1e6):
                mask_l = np.abs(length_flat[mask]*1e6 - l) < 0.01
                ax.plot(x_vals[mask_l], y_vals[mask_l])
            ax.set_title(f'{y_axis_name} vs {x_axis_name} (Lengths {np.min(unique_lengths_in_micro):.2f}-{np.max(unique_lengths_in_micro):.2f} μm)')
        else:
            ax.plot(x_vals, y_vals)
            ax.set_title(f'{y_axis_name} vs {x_axis_name} for {selected_length}')

        ax.set_xlabel(x_axis_name)
        ax.set_ylabel(y_axis_name)
        ax.grid(True)

        if log:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[], numticks=10))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'$10^{int(np.log10(x))}$'))

        # Determine active variable and index
        if active_var == 'X':
            idx = np.abs(x_vals - x_val).argmin()
        elif active_var == 'Y':
            idx = np.abs(y_vals - y_val).argmin()
        else:  # Z
            idx = np.abs(z_vals - z_val).argmin()

        # Scatter the selected point
        ax.scatter(x_vals[idx], y_vals[idx], color='red', zorder=5)

        # Pretty legend entries (always show X, Y, Z)
        legend_entries = [
            f"{x_axis_name}: {format_with_si(x_vals[idx])}",
            f"{y_axis_name}: {format_with_si(y_vals[idx])}",
            f"Vgs: {format_with_si(z_vals[idx])}"
        ]

        # Create invisible handles (so legend doesn't disappear)
        handles = [plt.Line2D([0], [0], color='none') for _ in legend_entries]

        ax.legend(handles, legend_entries,
                loc='best', frameon=True, fancybox=True,
                fontsize=12, framealpha=0.9,
                prop={'family': 'monospace'})


        # Update widget values (raw, not rounded)
        with x_box.hold_trait_notifications(), y_box.hold_trait_notifications(), z_box.hold_trait_notifications():
            x_box.value = format_for_box(x_vals[idx])
            y_box.value = format_for_box(y_vals[idx])
            z_box.value = format_for_box(z_vals[idx])

        plt.show()
        plt.close(fig)

        return None  # prevent interactive_output from displaying anything extra

    # Widgets
    dropdown_options = ["Show All"] + [f'{l:.2f} μm' for l in unique_lengths_in_micro]
    length_widget = widgets.Dropdown(options=dropdown_options, value=dropdown_options[0], description='Length:', layout=widgets.Layout(width='400px'))

    active_var_widget = widgets.RadioButtons(
        options=['X', 'Y', 'Z'],
        description='Control:',
        layout=widgets.Layout(width='200px')
    )

    x_box = widgets.FloatText(value=format_for_box(np.mean(x_values_flat)), description=f'{x_axis_name}:', layout=widgets.Layout(width='250px'))
    y_box = widgets.FloatText(value=format_for_box(np.mean(y_values_flat)), description=f'{y_axis_name}:', layout=widgets.Layout(width='250px'))
    z_box = widgets.FloatText(value=format_for_box(np.mean(z_values_flat)), description='Vgs:', layout=widgets.Layout(width='250px'))


    def toggle_active(change):
        x_box.disabled = change['new'] != 'X'
        y_box.disabled = change['new'] != 'Y'
        z_box.disabled = change['new'] != 'Z'

    active_var_widget.observe(toggle_active, names='value')

    out = interactive_output(update_plot, {
        'selected_length': length_widget,
        'active_var': active_var_widget,
        'x_val': x_box,
        'y_val': y_box,
        'z_val': z_box
    })

    display(VBox([
        length_widget,
        active_var_widget,
        HBox([x_box, y_box, z_box]),
        out
    ]))


def tile_length_to_match_data(length_array, data_array):
    length_array = np.array(length_array).flatten()
    data_shape = data_array.shape
    if length_array.size == data_shape[0]:
        return np.tile(length_array.reshape(-1, 1), (1, data_shape[1]))
    elif length_array.size == data_shape[1]:
        return np.tile(length_array.reshape(1, -1), (data_shape[0], 1))
    else:
        raise ValueError(f"Length array size {length_array.size} does not match data shape {data_shape}")
