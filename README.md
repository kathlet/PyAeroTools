# PyAeroTools
A Python Aerodynamic Analysis Toolkit

This repository contains a collection of Python tools for classic aerodynamic analysis, including a 2D vortex panel method, a 3D lifting-line theory solver, and a 2D potential flow simulator.

This project was developed to apply and demonstrate fundamental concepts in aerodynamics.

## Features

* **2D Vortex Panel Method (`panel_method_2d.py`)**
    * Analyzes 2D airfoils using a linear vortex panel method.
    * Supports custom airfoil coordinates from a file or automatic generation of NACA 4-digit airfoils.
    * Calculates and plots pressure coefficient ($C_p$) distribution.
    * Calculates and plots streamlines.
    * Performs an angle-of-attack sweep to generate $C_L$ vs. $\alpha$ curves.

* **3D Lifting-Line Theory (`lifting_line.py`)**
    * Models a 3D finite wing using classical lifting-line theory (LLT).
    * Uses a Fourier series to solve for the spanwise lift distribution.
    * Calculates total lift, induced drag, and rolling/yawing moments.
    * Accounts for wing planform, washout, aileron deflection, and roll rate.

* **2D Potential Flow Simulator (`flow_field.py`)**
    * Simulates a 2D potential flow field by superimposing elementary flows.
    * Supports freestream, sources/sinks, vortices, and doublets.
    * Plots the resulting streamlines.

## Generated Plots

(Run your scripts and save the plots. This is critical!)

#### Panel Method: NACA 2412 at 4° Alpha
!(examples/plots/naca2412_streamlines.png)
!(examples/plots/naca2412_cp.png)

#### Lifting-Line Theory: Elliptical Wing
!(examples/plots/llt_lift_distribution.png)

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/kathlet/PyAeroTools.git](https://github.com/kathlet/PyAeroTools.git)
    cd PyAeroTools
    ```

2.  (Recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Examples

All examples are located in the `/examples` directory.

### 1. Run the 2D Panel Method

Modify `examples/a26_input.json` to set parameters, then run:

```bash
python examples/run_panel_method.py
