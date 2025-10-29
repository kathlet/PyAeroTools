import os
import sys

# Ensure repo root is on sys.path so `aero_tools` can be imported when running
# this example directly (python examples/run_panel_method.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aero_tools.panel_method_2d import VortexPanelAirfoil

def main():
    airfoil = VortexPanelAirfoil("./examples/panel_method_input.json")
    if airfoil.plot_streamlines:
        airfoil.plot_airfoil()

    if airfoil.plot_pressure:
        airfoil.plot_surf_pressure()

    if airfoil.alpha_sweep:
        airfoil.alphaSweep()

    if airfoil.export_geometry:
        airfoil.exportGeometry()

if __name__ == "__main__":
    main()