import numpy as np
import matplotlib.pyplot as plt

def plot_flow_field(pfo):
    """Plot the flow field and streamlines."""
    fig, ax = plt.subplots()
    y_start_vals = np.linspace(-pfo.n_lines * pfo.delta_y, pfo.n_lines * pfo.delta_y, 2 * pfo.n_lines + 1) # Y values for the streamlines

    # Plot the streamlines
    for y_start in y_start_vals:
        streamline = pfo.streamline([pfo.x_start, y_start], pfo.delta_s)
        ax.plot(streamline[:, 0], streamline[:, 1], 'k')

    # Plot the elements
    for element in pfo.elements.values():
        x, y = element.get("x", 0), element.get("y", 0) 
        magnitude = element.get("velocity", element.get("lambda", element.get("gamma", element.get("kappa", 0))))
        color = 'blue' if magnitude > 0 else 'red' # Negative values are red, positive values are blue

        # Base the marker on the element type
        if element["type"] == "vortex":
            marker = 'x'
        elif element["type"] == "source":
            marker = '*'
        elif element["type"] == "doublet":
            marker = 'D'
        else:
            continue # Skip freestream elements (no marker)

        ax.scatter(x, y, color=color, s=20, marker=marker)

    ax.set_xlim(pfo.x_lower_limit, pfo.x_upper_limit)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Potential Flow Streamlines')
    ax.set_aspect('equal', 'box')
    plt.show()