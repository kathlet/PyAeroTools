import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class FlowFieldSimulator:
    def __init__(self, json_file_path):
        # Read and parse the JSON file
        json_string = open(json_file_path).read()
        json_vals = json.loads(json_string)

        # Extract values from the JSON object
        self.x_lower_limit = json_vals["plot"]["x_lower_limit"]
        self.x_upper_limit = json_vals["plot"]["x_upper_limit"]
        self.x_start = json_vals["plot"]["x_start"]
        self.delta_s = json_vals["plot"]["delta_s"]
        self.n_lines = json_vals["plot"]["n_lines"]
        self.delta_y = json_vals["plot"]["delta_y"]
        
        self.elements = json_vals["elements"]

    def velocity_field(self, point):
        x, y = point
        u, v = 0.0, 0.0  # Initialize velocity components
        for element in self.elements.values():
            if element["type"] == "freestream":
                velocity = element["velocity"]
                alpha = np.radians(element["angle_of_attack"])
                u += velocity * np.cos(alpha)
                v += velocity * np.sin(alpha)

            elif element["type"] == "source":
                lambda_ = element["lambda"]
                x0, y0 = element["x"], element["y"]
                x_s = x - x0
                y_s = y - y0
                u += lambda_ / (2 * np.pi) * x_s / (x_s**2 + y_s**2)
                v += lambda_ / (2 * np.pi) * y_s / (x_s**2 + y_s**2)
                
            elif element["type"] == "vortex":
                gamma = element["gamma"]
                x0, y0 = element["x"], element["y"]
                x_s = x - x0
                y_s = y - y0
                u += gamma / (2 * np.pi) * y_s / (x_s**2 + y_s**2)
                v += -gamma / (2 * np.pi) * x_s / (x_s**2 + y_s**2)
                
            elif element["type"] == "doublet":
                kappa = element["kappa"]
                x0, y0 = element["x"], element["y"]
                x_s = x - x0
                y_s = y - y0
                u += -kappa / (2 * np.pi) * (x_s**2 - y_s**2) / (x_s**2 + y_s**2)**2
                v += -kappa / (2 * np.pi) * 2 * x_s * y_s / (x_s**2 + y_s**2)**2
        
        return np.array([u, v])

    def streamline(self, start, delta_s):
        def velocity_field_wrapper(s, pos):
            velocity = self.velocity_field(pos)
            norm = np.linalg.norm(velocity)
            return velocity/norm if norm > 0 else velocity # Normalize the velocity vector, This ensures an equal step size along the streamline

        def rk4_step(func, s, pos, delta_s):
            k1 = delta_s * np.array(func(s, pos))
            k2 = delta_s * np.array(func(s + delta_s / 2, pos + k1 / 2))
            k3 = delta_s * np.array(func(s + delta_s / 2, pos + k2 / 2))
            k4 = delta_s * np.array(func(s + delta_s, pos + k3))

            return pos + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        streamline_array = []
        pos = np.array(start)
        s = 0

        while self.x_lower_limit <= pos[0] <= self.x_upper_limit:
            streamline_array.append(pos.copy())  # Save the current position
            pos = rk4_step(velocity_field_wrapper, s, pos, delta_s) # Compute the next position
            s += delta_s  # Increment along the path length

        return np.array(streamline_array)




