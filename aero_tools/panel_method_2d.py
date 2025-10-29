import json
import numpy as np
import matplotlib.pyplot as plt

class VortexPanelAirfoil:
    def __init__(self, json_file_path):
        # Read and parse the JSON file
        json_string = open(json_file_path).read()
        json_vals = json.loads(json_string)

        print("\nReading JSON file...")
        # Extract values from the JSON object
        # Operating conditions
        self.v_inf = json_vals["operating"]["freestream_velocity"]
        self.alpha = np.radians(json_vals["operating"]["alpha[deg]"])
        # Alpha sweep values
        self.output_file = json_vals["alpha_sweep"]["output_file"]
        self.start = np.radians(json_vals["alpha_sweep"]["start[deg]"])
        self.end = np.radians(json_vals["alpha_sweep"]["end[deg]"])
        self.increment = np.radians(json_vals["alpha_sweep"]["increment[deg]"])
        # Plot options
        self.x_lower_limit = json_vals["plot_options"]["x_lower_limit"]
        self.x_upper_limit = json_vals["plot_options"]["x_upper_limit"]
        self.x_start = json_vals["plot_options"]["x_start"]
        self.delta_s = json_vals["plot_options"]["delta_s"]
        self.n_lines = json_vals["plot_options"]["n_lines"]
        self.delta_y = json_vals["plot_options"]["delta_y"]
        # Run commands
        self.plot_streamlines = json_vals["run_commands"]["plot_streamlines"]
        self.plot_pressure = json_vals["run_commands"]["plot_pressure"]
        self.alpha_sweep = json_vals["run_commands"]["alpha_sweep"]
        self.export_geometry = json_vals["run_commands"]["export_geometry"]
        self.CL_design = json_vals["geometry"]["CL_design"]
        self.trailing_edge = json_vals["geometry"]["trailing_edge"]
        self.airfoil = json_vals["geometry"]["airfoil"]

        if self.airfoil == "file":
            self.filename = json_vals["geometry"]["filename"]
            print("Reading airfoil geometry from file: ", self.filename)
            # Read the airfoil geometry from the specified file that has x in column 1 and y in column 2
            data = np.loadtxt(self.filename, skiprows=0)
            self.x_points = data[:, 0]
            self.y_points = data[:, 1]
            self.n_points = len(self.x_points)
            # Calculate the x_coords_camber and y_coords_camber by taking the midpoint
            self.x_coords_camber = np.zeros(self.n_points//2)
            self.y_coords_camber = np.zeros(self.n_points//2)

            for i in range(self.n_points//2):
                j = self.n_points - i - 1
                self.x_coords_camber[i] = 0.5 * (self.x_points[i] + self.x_points[j])
                self.y_coords_camber[i] = 0.5 * (self.y_points[i] + self.y_points[j])
                                                                                  
        else:
            self.n_points = json_vals["geometry"]["n_points"]
            # Generate the NACA airfoil points
            self.x_upper, self.y_upper, self.x_lower, self.y_lower, self.x_coords_camber, self.y_coords_camber = self.generate_naca_points()
            # Generate the control points and calculate the A matrix
            self.x_points = np.concatenate((self.x_lower, self.x_upper))
            self.y_points = np.concatenate((self.y_lower, self.y_upper))

        # Calculate the control point lengths and midpoints
        self.n = len(self.x_points) - 1
        self.l = np.zeros(self.n)
        for i in range(self.n):
            self.l[i] = np.sqrt((self.x_points[i + 1] - self.x_points[i]) ** 2 + (self.y_points[i + 1] - self.y_points[i]) ** 2)
        self.x_c = np.zeros(self.n)
        self.y_c = np.zeros(self.n)
        for i in range(self.n):
            self.x_c[i] = (self.x_points[i] + self.x_points[i + 1]) / 2
            self.y_c[i] = (self.y_points[i] + self.y_points[i + 1]) / 2
        self.A_matrix = self.Amatrix()
        self.gamma_array = self.gamma(self.alpha)
        
    def geometry(self, x_over_c):
        # Check if airfoil is ULxx (uniform-load camber line) or NACA 4-digit
        if "UL" in self.airfoil:
            # For ULxx airfoil, extract thickness from the last two digits (xx)
            t = int(self.airfoil[2:]) / 100.0  # Maximum thickness
    
            # Use uniform-load camber line (NACA 1-series)
            CLd = self.CL_design
        
            if np.isclose(x_over_c, 0):
                yc = 0
                dyc_dx = 0
            elif np.isclose(x_over_c, 1):
                yc = 0
                dyc_dx = 0
            else:
                yc = CLd / (4 * np.pi) * ((x_over_c - 1) * np.log(1 - x_over_c) - x_over_c * np.log(x_over_c))
                dyc_dx = CLd / (4 * np.pi) * (np.log(1 - x_over_c) - np.log(x_over_c))
        else:
            # Use traditional NACA 4-digit camber line
            m = int(self.airfoil[0]) / 100.0  # Maximum camber
            p = int(self.airfoil[1]) / 10.0   # Position of maximum camber
            t = int(self.airfoil[2:]) / 100.0 # Maximum thickness
    
            if m == 0:
                yc = 0
                dyc_dx = 0
            else:
                if x_over_c <= p:
                    yc = m / p**2 * (2 * p * x_over_c - x_over_c**2)
                    dyc_dx = 2 * m / p**2 * (p - x_over_c)
                else:
                    yc = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x_over_c - x_over_c**2)
                    dyc_dx = 2 * m / (1 - p)**2 * (p - x_over_c)
    
        # Calculate the thickness distribution (yt) based on the trailing edge type
        if self.trailing_edge == "open":
            yt = 5 * t * (0.2969 * np.sqrt(x_over_c) - 0.1260 * x_over_c - 0.3516 * x_over_c**2 +
                          0.2843 * x_over_c**3 - 0.1015 * x_over_c**4)
        elif self.trailing_edge == "closed":
            yt = t / 2 * (2.980 * np.sqrt(x_over_c) - 1.320 * x_over_c - 3.286 * x_over_c**2 +
                          2.441 * x_over_c**3 - 0.815 * x_over_c**4)
    
        # Calculate the angle of the camber line
        theta_camber = np.arctan(dyc_dx)
    
        # Calculate upper and lower surface points
        x_cam = x_over_c
        y_cam = yc
        x_u = x_over_c - yt * np.sin(theta_camber)
        y_u = yc + yt * np.cos(theta_camber)
        x_l = x_over_c + yt * np.sin(theta_camber)
        y_l = yc - yt * np.cos(theta_camber)
    
        camber = np.array([x_cam, y_cam])
        upper_surface = np.array([x_u, y_u])
        lower_surface = np.array([x_l, y_l])
    
        return camber, upper_surface, lower_surface

    def generate_naca_points(self):
        print("Generating NACA points...")
        x_coords_upper = []
        y_coords_upper = []
        x_coords_lower = []
        y_coords_lower = []
        x_coords_camber = []
        y_coords_camber = []
        half_points = self.n_points // 2        
        # Odd number of nodes
        if self.n_points % 2 == 1:
            delta_theta = np.pi / half_points
            x_coords_upper.append(0.0)
            y_coords_upper.append(0.0)
            for i in range(1, half_points + 1):
                theta = i * delta_theta
                x_over_c = 0.5 * (1 - np.cos(theta))  # x/c
                camber, upper_surface, lower_surface = self.geometry(x_over_c)
                x_cam, y_cam = camber
                x_u, y_u = upper_surface
                x_l, y_l = lower_surface
                x_coords_upper.append(x_u)
                y_coords_upper.append(y_u)
                x_coords_lower.append(x_l)
                y_coords_lower.append(y_l)
                x_coords_camber.append(x_cam)
                y_coords_camber.append(y_cam)
            x_coords_lower.reverse()
            y_coords_lower.reverse()
            x_coords_camber.insert(0,0.0)
            y_coords_camber.insert(0,0.0)
        else:
            delta_theta = np.pi / (half_points - 0.5)
            for i in range(1, half_points + 1):
                theta = i * delta_theta - 0.5 * delta_theta
                x_over_c = 0.5 * (1 - np.cos(theta))  # x/c
                camber, upper_surface, lower_surface = self.geometry(x_over_c)
                x_cam, y_cam = camber
                x_u, y_u = upper_surface
                x_l, y_l = lower_surface
                x_coords_upper.append(x_u)
                y_coords_upper.append(y_u)
                x_coords_lower.append(x_l)
                y_coords_lower.append(y_l)
                x_coords_camber.append(x_cam)
                y_coords_camber.append(y_cam)
            x_coords_lower.reverse()
            y_coords_lower.reverse()
        return x_coords_upper, y_coords_upper, x_coords_lower, y_coords_lower, x_coords_camber, y_coords_camber

    def surface_normal(self, x):
        delta = 1e-7

        if (x) < (self.leading_edge_x + delta):  # Near the leading edge
            _, upper_surface, lower_surface = self.geometry(x + delta)
            upper_surface_tangent = np.array(upper_surface) - np.array(lower_surface)
            lower_surface_tangent = np.array(upper_surface) - np.array(lower_surface)
            upper_surface_normal = np.array([-upper_surface_tangent[1], upper_surface_tangent[0]])
            lower_surface_normal = np.array([-lower_surface_tangent[1], lower_surface_tangent[0]])
            upper_surface_normal /= np.linalg.norm(upper_surface_normal)
            lower_surface_normal /= np.linalg.norm(lower_surface_normal)
        elif (x) > (self.trailing_edge_x - delta):  # Near the trailing edge
            _, _, lower_surface = self.geometry(x - delta)
            _, upper_surface, _ = self.geometry(x - delta)
            upper_surface_tangent = np.array(upper_surface) - np.array(lower_surface)
            lower_surface_tangent = np.array(upper_surface) - np.array(lower_surface)
            upper_surface_normal = np.array([-upper_surface_tangent[1], upper_surface_tangent[0]])
            lower_surface_normal = np.array([-lower_surface_tangent[1], lower_surface_tangent[0]])
            upper_surface_normal /= -np.linalg.norm(upper_surface_normal)
            lower_surface_normal /= -np.linalg.norm(lower_surface_normal)
        else:  # Central difference for points not near edges
            _, upper_surface_before, lower_surface_before = self.geometry(x - delta)
            _, upper_surface_after, lower_surface_after = self.geometry(x + delta)
            upper_surface_tangent = (np.array(upper_surface_after) - np.array(upper_surface_before))
            lower_surface_tangent = (np.array(lower_surface_after) - np.array(lower_surface_before))
            upper_surface_normal = np.array([-upper_surface_tangent[1], upper_surface_tangent[0]])
            lower_surface_normal = np.array([-lower_surface_tangent[1], lower_surface_tangent[0]])
            upper_surface_normal /= np.linalg.norm(upper_surface_normal)
            lower_surface_normal /= -np.linalg.norm(lower_surface_normal)

        return upper_surface_normal, lower_surface_normal

    def surface_tangent(self, x, delta=1e-8):
        if x < self.leading_edge_x + delta:
            # Near the leading edge, use forward difference
            _, upper_surface_1, lower_surface_1 = self.geometry(x)
            _, upper_surface_2, lower_surface_2 = self.geometry(x + delta)
        elif x > self.trailing_edge_x - delta:
            # Near the trailing edge, use backward difference
            _, upper_surface_1, lower_surface_1 = self.geometry(x - delta)
            _, upper_surface_2, lower_surface_2 = self.geometry(x)
        else:
            # Use central difference
            _, upper_surface_1, lower_surface_1 = self.geometry(x - delta)
            _, upper_surface_2, lower_surface_2 = self.geometry(x + delta)
    
        # Calculate the tangent vectors
        upper_surface_tangent = np.array(upper_surface_2) - np.array(upper_surface_1)
        lower_surface_tangent = np.array(lower_surface_2) - np.array(lower_surface_1)

        # Normalize the tangent vectors
        upper_surface_tangent /= np.linalg.norm(upper_surface_tangent)
        lower_surface_tangent /= -np.linalg.norm(lower_surface_tangent)

        return upper_surface_tangent, lower_surface_tangent

    def P(self, j, i, x, y):
        # Compute the normal vector xi and eta
        Matrix = np.array([[self.x_points[j + 1] - self.x_points[j], self.y_points[j + 1] - self.y_points[j]], [self.y_points[j] - self.y_points[j + 1], self.x_points[j + 1] - self.x_points[j]]])
        OtherMatrix = np.array([[x[i] - self.x_points[j]], [y[i] - self.y_points[j]]])
        [[xi], [eta]] = 1/self.l[j]*np.matmul(Matrix, OtherMatrix)

        # find phi and psi
        phi = np.zeros(self.n)
        psi = np.zeros(self.n)
        if np.isclose(xi, self.l[j], atol=1e-8) and np.isclose(eta, 0, atol=1e-8):
            psi = 0  # Or any appropriate value depending on the problem
        else:
            psi = 1/2 * np.log(((xi**2 + eta**2) / (eta**2 + (xi - self.l[j])**2) + 1e-12))
        phi = np.arctan2(eta*self.l[j], eta**2 + xi**2 - xi*self.l[j])
        #Develope P matrix
        P1matrix = [[(self.x_points[j+1]-self.x_points[j]), -(self.y_points[j+1]-self.y_points[j])], [(self.y_points[j+1]-self.y_points[j]), (self.x_points[j+1]-self.x_points[j])]]
        P2matrix = [[(self.l[j]-xi)*phi+eta*psi, (xi*phi-eta*psi)], [eta*phi-(self.l[j]-xi)*psi-self.l[j], (-eta*phi-xi*psi+self.l[j])]]
        P = 1/(2*np.pi*self.l[j]**2)*np.matmul(P1matrix, P2matrix)
    
        return P
    
    def Amatrix(self):
        A = np.zeros((self.n+1,self.n+1))
        for i in range(0,self.n):
            for j in range(0,self.n):
                P_matrix = self.P(j, i, self.x_c, self.y_c)
                
                A[i,j] += ((self.x_points[i+1] - self.x_points[i]) / self.l[i]) * P_matrix[1,0] - ((self.y_points[i+1] - self.y_points[i]) / self.l[i]) * P_matrix[0,0]
                A[i,j+1] += ((self.x_points[i+1] - self.x_points[i]) / self.l[i]) * P_matrix[1,1] - ((self.y_points[i+1] - self.y_points[i]) / self.l[i]) * P_matrix[0,1]

        # Enforce the Kutta condition
        A[self.n,0] = 1.0
        A[self.n,self.n] = 1.0
        
        return A
    
    def gamma(self, alpha):
        B = np.zeros(self.n+1)
        for i in range(0,self.n):
            B[i] = self.v_inf*((self.y_points[i+1]-self.y_points[i])*np.cos(alpha) - (self.x_points[i+1]-self.x_points[i])*np.sin(alpha))/self.l[i]
        B[self.n] = 0.0
        A = self.A_matrix
        gamma = np.linalg.solve(A,B)
        return gamma
 
    def velocity_field(self, point):
        Vx = self.v_inf * np.cos(self.alpha)
        Vy = self.v_inf * np.sin(self.alpha)

        x_point, y_point = point
        x_point = np.array([x_point]) # convert each point to a array
        y_point = np.array([y_point])

        vxsum = 0.0
        vysum = 0.0
        for i in range(0,self.n):
            P_matrix = self.P(i, 0, x_point, y_point)
            gamma_i = self.gamma_array[i]
            gamma_iplus1 = self.gamma_array[i+1]
            vxsum += P_matrix[0,0]*gamma_i + P_matrix[0,1]*gamma_iplus1
            vysum += P_matrix[1,0]*gamma_i + P_matrix[1,1]*gamma_iplus1
        Vx += vxsum
        Vy += vysum
        return Vx, Vy

    def surface_tangential_velocity(self, x):   
        # Get the points on the upper and lower surfaces
        point_upper = np.array([self.geometry(x)[1][0], self.geometry(x)[1][1]])
        point_lower = np.array([self.geometry(x)[2][0], self.geometry(x)[2][1]])

        # Get the velocity vectors at the upper and lower surface points
        v_upper = self.velocity_field(point_upper)
        v_lower = self.velocity_field(point_lower)
        
        # Get the tangent vectors at the upper and lower surface points
        upper_surface_tangent, lower_surface_tangent = self.surface_tangent(x)
        
        # Project the velocity vectors onto the tangent vectors to get the tangential velocity magnitudes
        upper_tangential_velocity = np.dot(v_upper, upper_surface_tangent)
        lower_tangential_velocity = np.dot(v_lower, lower_surface_tangent)

        return upper_tangential_velocity, lower_tangential_velocity

    def streamline(self, start, delta_s):
        def velocity_field_wrapper(s, pos): # Wrapper for the velocity field function
            norm = np.linalg.norm(self.velocity_field(pos))
            if norm == 0:
                return self.velocity_field(pos) + 1e-6 # if the velocity is zero, add a small value to allow the streamline to move in a direction
            return self.velocity_field(pos)/norm
    
        def rk4_step(func, s, pos, delta_s): # Runge-Kutta 4th order method
            k1 = delta_s * np.array(func(s, pos))
            k2 = delta_s * np.array(func(s + delta_s/2, pos + k1/2))
            k3 = delta_s * np.array(func(s + delta_s/2, pos + k2/2))
            k4 = delta_s * np.array(func(s + delta_s, pos + k3))
    
            return pos + (k1 + 2*k2 + 2*k3 + k4) / 6

        # Initialize streamline array
        streamline_array = []
        pos = np.array(start)
        s = 0 # Streamline parameter

        # Integrate until reaching x-limits
        while self.x_lower_limit <= pos[0] <= self.x_upper_limit:
            streamline_array.append(pos.copy())  # Save the current position
            pos = rk4_step(velocity_field_wrapper, s, pos, delta_s)  # Perform RK4 step
            s += delta_s  # Increment the streamline parameter
            # Loading bar indicator to the terminal based on percentage of streamline completion pos
            percentage = (pos[0] - self.x_lower_limit) / (self.x_upper_limit - self.x_lower_limit) * 100
            # using blocks to create the loading bar
            blocks = int(percentage // 2)
            spaces = int(50 - blocks)
            print(f"\r[{'#' * blocks}{' ' * spaces}] {percentage:.0f}%", end="")
        print("\n")
        return np.array(streamline_array)

    def stagnation(self):
        # Define tolerance for checking near-zero velocities
        tol = 1e-10 * self.v_inf
        max_iter = 10000

        self.leading_edge_x = 0 # self.x_coords_camber[0]
        self.trailing_edge_x = 1.01 # self.x_coords_camber[-1]

        # Bisection method to find stagnation point
        def find_stagnation_point_bisection(x_left, x_right, surface_check):
            for _ in range(max_iter):
                x_mid = (x_left + x_right) / 2.0
                
                # Compute tangential velocities at left, right, and midpoint
                upper_tangential_velocity_left, lower_tangential_velocity_left = self.surface_tangential_velocity(x_left)
                upper_tangential_velocity_right, lower_tangential_velocity_right = self.surface_tangential_velocity(x_right)
                upper_tangential_velocity_mid, lower_tangential_velocity_mid = self.surface_tangential_velocity(x_mid)

                # Select the appropriate surface (upper or lower)
                velocity_left = upper_tangential_velocity_left if surface_check == 'upper' else lower_tangential_velocity_left
                velocity_right = upper_tangential_velocity_right if surface_check == 'upper' else lower_tangential_velocity_right
                velocity_mid = upper_tangential_velocity_mid if surface_check == 'upper' else lower_tangential_velocity_mid
                
                # Check if the midpoint tangential velocity is close to zero (stagnation point found)
                if abs(velocity_mid) < tol:
                    print("Stagnation point found...")
                    return x_mid
                # Determine which side of the midpoint contains the root
                if velocity_left * velocity_mid < 0:
                    x_right = x_mid  # Root is between x_left and x_mid
                else:
                    x_left = x_mid  # Root is between x_mid and x_right

            raise ValueError("Bisection method did not converge within the maximum number of iterations")

        # Check at the leading edge
        print("Checking for stagnation at the leading edge")
        upper_tangential_velocity, lower_tangential_velocity = self.surface_tangential_velocity(self.leading_edge_x)
    
        if abs(upper_tangential_velocity) < tol:
            print("Stagnation point found at the leading edge")
            forward_stagnation_point = (self.leading_edge_x, self.geometry(self.leading_edge_x)[1][1])
            print("\nchordwise x/c value =                           ", self.leading_edge_x)
            xoverc_forward_stagnation = self.leading_edge_x
        elif lower_tangential_velocity < 0 and upper_tangential_velocity > 0:
            print("Stagnation point found at the leading edge (upper surface)")
            forward_stagnation_point = (self.leading_edge_x, self.geometry(self.leading_edge_x)[2][1])
            print("\nchordwise x/c value =                           ", self.leading_edge_x)
            xoverc_forward_stagnation = self.leading_edge_x
        elif upper_tangential_velocity > 0:
            print("Stagnation point not found at the leading edge (upper surface)")
            # Positive velocity, search the lower surface for stagnation point using bisection
            x_lower_stagnation = find_stagnation_point_bisection(self.leading_edge_x, self.leading_edge_x + 0.1, 'lower')
            forward_stagnation_point = (self.geometry(x_lower_stagnation)[2][0], self.geometry(x_lower_stagnation)[2][1])
            print("\nchordwise x/c value =                           ", x_lower_stagnation)
            xoverc_forward_stagnation = x_lower_stagnation
        else:
            print("Stagnation point not found at the leading edge (lower surface)")
            # Negative velocity, search the upper surface for stagnation point using bisection
            x_upper_stagnation = find_stagnation_point_bisection(self.leading_edge_x, self.leading_edge_x + 0.1, 'upper')
            forward_stagnation_point = (self.geometry(x_upper_stagnation)[1][0], self.geometry(x_upper_stagnation)[1][1])
            print("\nchordwise x/c value =                           ", x_upper_stagnation)
            xoverc_forward_stagnation = x_upper_stagnation
    
        print(f"stagnation point (x/c, y/c) =                    ({float(forward_stagnation_point[0]):.10e}, {float(forward_stagnation_point[1]):.10e})")
        vel = self.velocity_field(forward_stagnation_point)
        print(f"velocity at stagnation point [ft/s] =            ({float(vel[0]):.6f}, {float(vel[1]):.10e})")
        print(f"pressure coefficient at stagnation point =        {float(self.pressure_coefficient(*forward_stagnation_point)):.10e}")
        vx, vy = self.surface_tangential_velocity(xoverc_forward_stagnation)
        if vx < vy:
            stag_vel = vy
        else:
            stag_vel = vx
        print(f"tangential velocity at stagnation point [ft/s] = ({float(stag_vel):.6e})\n")

        # print("Forward Stagnation Point: ", forward_stagnation_point)
        print("Checking for stagnation at the trailing edge")
        # Set trailing edge as the stagnation point
        aft_stagnation_point = (self.trailing_edge_x, self.y_coords_camber[-1])
        # print("Aft Stagnation Point: ", aft_stagnation_point)
        return forward_stagnation_point, aft_stagnation_point
        
    def pressure_coefficient(self, x, y):
        # Compute the velocity at the point
        Vx, Vy = self.velocity_field([x, y])
        V = np.sqrt(Vx**2 + Vy**2)
        
        # Compute the pressure coefficient
        Cp = 1 - (V / self.v_inf)**2
        return Cp

    def plot_airfoil(self):
        # Plot the airfoil surfaces (lower, upper, and camber line)
        x = self.x_points
        y = self.y_points
        plt.plot(x, y, 'b-', label='Airfoil')
        plt.plot(self.x_coords_camber, self.y_coords_camber, 'r-', label='Camber Line')        

        if self.airfoil == "file":
            # set the forward stagnation point to (-0.01, 0) and aft stagnation point to (1.00, 0)
            forward_stagnation_point = (-0.01, 0)
            aft_stagnation_point = (1.01, 0)
        else:
            # Calculate stagnation points
            forward_stagnation_point, aft_stagnation_point = self.stagnation() 

        # Plot the forward and aft stagnation streamlines
        plt.plot(*zip(*self.streamline(forward_stagnation_point, -self.delta_s)), color='k')
        plt.plot(*zip(*self.streamline(aft_stagnation_point, self.delta_s)), color='k')

        # Get y at minimum x value from forward streamline
        streamline_forward = self.streamline(forward_stagnation_point, -self.delta_s)
        y_at_min_x = streamline_forward[np.argmin(streamline_forward[:, 0]), 1]

        # Generate streamlines around the forward stagnation point
        y_min, y_max = y_at_min_x - self.n_lines * self.delta_y, y_at_min_x + self.n_lines * self.delta_y
        y_values = np.arange(y_min, y_max + self.delta_y, self.delta_y)
        
        print("\nPlotting streamlines...")

        # Plot additional streamlines
        for y in y_values:
            print("Plotting streamline at y = ", y)
            if abs(y - y_at_min_x) < self.delta_y / 2:
                continue
            streamline = self.streamline([self.x_start, y], self.delta_s)
            plt.plot(streamline[:, 0], streamline[:, 1], color='k')

        ax = plt.gca()
        ax.set_aspect('equal')
        plt.title(f"NACA {self.airfoil} Streamlines")

        # Set axis limits and display the plot
        plt.xlim(self.x_lower_limit, self.x_upper_limit)
        plt.ylim(y_min, y_max)
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.show()

    def plot_surf_pressure(self):
        """Plot the pressure distribution on the airfoil surface. The pressure coefficient (Cp)
        is calculated at points slightly off the control points along the surface normal to avoid 
        sharp tangential velocity changes at the control points."""

        self.leading_edge_x = 0 # self.x_coords_camber[0]
        self.trailing_edge_x = -1 # self.x_coords_camber[-1]

        Cp_upper = []
        x_graph_upper = []

        # Avoid using the first and last control points directly
        for i in range(self.n):
            x = self.x_c[i]
            y = self.y_c[i]

            # Calculate tangent vector using adjacent points (central difference for internal points)
            if i == 0:
                # Forward difference for trailing edge (first point)
                dx = self.x_c[i+1] - self.x_c[i]
                dy = self.y_c[i+1] - self.y_c[i]
            elif i == self.n - 1:
                # Backward difference for trailing edge (last point)
                dx = self.x_c[i] - self.x_c[i-1]
                dy = self.y_c[i] - self.y_c[i-1]
            else:
                # Central difference for interior points
                dx = self.x_c[i+1] - self.x_c[i-1]
                dy = self.y_c[i+1] - self.y_c[i-1]

            # Normalize the tangent vector
            tangent = np.array([dx, dy])
            tangent /= np.linalg.norm(tangent)

            # Calculate normal vector (perpendicular to tangent)
            normal = np.array([-tangent[1], tangent[0]])

            # Sample points slightly off the surface in the normal direction
            point_upper = np.array([x, y]) + 1e-16  * normal

            # Compute pressure coefficients
            cp_upper = self.pressure_coefficient(*point_upper)

            Cp_upper.append(cp_upper)
            x_graph_upper.append(x)

        # Print the lift coefficient
        def cl_streamline():
            # Calculate the lift coefficient
            gamma = self.gamma(self.alpha)
            
            lift = 0.0
            for i in range(0,self.n):
                lift += self.l[i]*(gamma[i] + gamma[i+1])/self.v_inf
            return lift

        print("\nLift Coefficient from Streamline Method: ", cl_streamline())

        # Plotting
        plt.plot(x_graph_upper, Cp_upper, 'b-', label='Upper Surface')
        plt.xlabel("x/c")
        plt.ylabel("Pressure Coefficient (Cp)")
        plt.title(f"Pressure Coefficient Distribution for Airfoil {self.airfoil} with lift coefficient {cl_streamline():.5f}")
        

        # Ensure Cp is positive and invert the y-axis for the plot
        plt.ylim(plt.ylim()[::-1])
        plt.grid(True)
        plt.show()
    
    def alphaSweep(self):
        def cl():
            # Calculate the lift coefficient
            gamma = self.gamma(alpha)
            
            lift = 0.0
            for i in range(0,self.n):
                lift += self.l[i]*(gamma[i] + gamma[i+1])/self.v_inf
            return lift
        def cmle(alpha):
            # Calculate the moment coefficient about the leading edge
            gamma = self.gamma(alpha)
            moment = 0.0
            for i in range(0,self.n):
                moment += (self.l[i]*(2*self.x_points[i]*gamma[i] + self.x_points[i]*gamma[i+1] + self.x_points[i+1]*gamma[i] + 2*self.x_points[i+1]*gamma[i+1])/self.v_inf * np.cos(alpha) + ((2*self.y_points[i]*gamma[i] + self.y_points[i]*gamma[i+1] + self.y_points[i+1]*gamma[i] + 2*self.y_points[i+1]*gamma[i+1])/self.v_inf) * np.sin(alpha))
            return moment * -1/3
        def cmc4(cmle):
            # Calculate the moment coefficient about the quarter chord
            
            cmc4 = cmle + cl()/4
            return cmc4
        
        print ("\nVortex Panel Method Results...")

        # Initialize arrays to store the results
        alphas = np.linspace(self.start, self.end, num=int((self.end - self.start) / self.increment) + 1)        
        cls = np.zeros_like(alphas)
        cmles = np.zeros_like(alphas)
        cmc4s = np.zeros_like(alphas)
        print("----------------------------------------------")
        print("----------------------------------------------")
        print("Alpha[deg]      Cl         Cmle       Cmc/4")
        print("----------------------------------------------")
        for i, alpha in enumerate(alphas):
            cls[i] = cl()
            cmles[i] = cmle(alpha)
            cmc4s[i] = cmc4(cmles[i])
            print(f"   {np.degrees(alpha):.0f}        {cls[i]:.5f}     {cmles[i]:.5f}    {cmc4s[i]:.5f}")
        print("----------------------------------------------")
        print("----------------------------------------------")

        # Output the results to self.output_file which is a csv file
        output_file = self.output_file
        with open(output_file, "w") as file:
            file.write("Alpha[deg],Cl,Cmle,Cmc/4\n")
            for i, alpha in enumerate(alphas):
                file.write(f"{np.degrees(alpha):.0f},{cls[i]:.5f},{cmles[i]:.5f},{cmc4s[i]:.5f}\n")


        # # print the lift, leading edge moment, and quarter chord moment at zero angle of attack to 14 decimal places
        # print("\nLift Coefficient at zero angle of attack: ", cls[6])
        # print("Leading Edge Moment Coefficient at zero angle of attack: ", cmles[6])
        # print("Quarter Chord Moment Coefficient at zero angle of attack: ", cmc4s[6])

    def exportGeometry(self):
        """The user can choose to write their airfoil geometry out to a text file in double precision."""
        print("Exporting geometry...")
        # Create a filename based on airfoil and number of points
        filename = f"{self.airfoil}_{self.n_points}.txt"

        # Open the file and write the data
        with open(filename, "w") as file:
            # Write x and y points to the file
            for i in range(self.n + 1):
                file.write(f"{self.x_points[i]:.16f}\t{self.y_points[i]:.16f}\n")

        print(f"Geometry exported to {filename}")


