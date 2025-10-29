"""Plots the planform and lift distribution components across the wing.
Number of points N is declared on line 12."""

import numpy as np
import matplotlib.pyplot as plt

# ZACHS CODE STARTS HERE

def interpolate(x1,x2,x,y1,y2):
    return (y2-y1)/(x2-x1)*(x-x1)+y1

def theta2s(theta):
    return -1.*np.cos(theta)

def s2theta(s):
    return np.arccos(-s)

def omega(th):
    return abs(np.cos(th))

def chi(th):
    s = theta2s(th)
    if s < -sat or abs(s) <= sar or sat < s:
        return 0.
    elif -sat <= s and s < -sar:
        return flapEfficiency(th)
    else:
        return -flapEfficiency(th)

def flapEfficiency(th):
    s = abs(theta2s(th))
    hs = interpolate(sar,sat,s,har,hat)
    c = chord(th)
    te = 0.75*c
    fcf = (te - hs) / c
    thf = np.arccos(2*fcf-1)
    ei = 1 - (thf - np.sin(thf)) / np.pi
    return de*he*ei

def chord(th):
    # s = theta2s(th)
    # cw = 1.
    # cr, ct = 2.*cw/(1.+Rt), 2.*Rt*cw/(1.+Rt)
    # return cr + abs(s) * (ct - cr)
    return 4 / np.pi * np.sin(th)

isElliptic = True ## flag that can be used to know if the planform is elliptic (useful when creating the C matrix)
Ra = 8
bw = Ra
cw = 1
Rt = 0.4
CLa = 2*np.pi

aoa_deg     = 5
Omega_deg   = -1
das_deg      = 7
pbar        = -0.1

sar     = 0.5
sat     = 0.9
fcf_ar  = 0.25
fcf_at  = 0.35
de      = 1.
he      = 0.85

car = chord(s2theta(sar))
cat = chord(s2theta(sat))
har = (0.75-fcf_ar)*car
hat = (0.75-fcf_at)*cat

# ZACHS CODE ENDS HERE

# Global parameters
N = 99                          # Number points
aoa = np.radians(aoa_deg)       # Angle of attack
Omega = np.radians(Omega_deg)   # Roll rate
das = np.radians(das_deg)       # Aileron deflection

def cosine_clustering():
    """Generate cosine clustered points for z/b."""
    theta = np.linspace(0, np.pi, N)
    return theta, theta2s(theta) / 2

def solve_fourier_coefficients(theta):
    """Solve for decomposed Fourier coefficients a_j, b_j, c_j, and d_j."""
    c_theta = chord(theta)
    if c_theta[0] == 0:
        c_theta[0] = 1e-10
    if c_theta[-1] == 0:
        c_theta[-1] = 1e-10
    omega_values = omega(theta)
    chi_values = np.array([chi(th) for th in theta])
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0 and isElliptic == True:
                C[i, j] = (np.pi * bw / CLa + (j + 1)) * (j + 1)
            elif i == 0 and isElliptic == False:
                C[i, j] = (j + 1) ** 2
            elif i == N - 1 and isElliptic == True:
                C[i, j] = (np.pi * bw / CLa + (j + 1)) * (j + 1) * (-1) ** (j + 2)
            elif i == N - 1 and isElliptic == False:
                C[i, j] = (-1) ** (j + 2) * (j + 1) ** 2
            else:
                C[i, j] = (4 * bw / (CLa * c_theta[i]) + (j + 1) / np.sin(theta[i])) * np.sin((j + 1) * theta[i])
    a = np.linalg.solve(C, np.ones(N))
    b = np.linalg.solve(C, omega_values)
    c = np.linalg.solve(C, chi_values)
    d = np.linalg.solve(C, np.cos(theta))
    return a, b, c, d

def A_matrix(a, b, c, d):
    """Calculate the A matrix for lift, drag, rolling moment, and yawing moment."""
    A = np.zeros(N)
    for i in range(N):
        if i % 2 == 0:
            A[i] = a[i] * aoa - b[i] * Omega
        else:
            A[i] = c[i] * das + d[i] * pbar
    return A

def calculate_lift_contributions(a, b, c, d):
    """Calculate the lift, drag, rolling moment, and yawing moment contributions of each component."""
    zeros = np.zeros(N)
    A_planform = A_matrix(a, zeros, zeros, zeros)
    A_washout = A_matrix(zeros, b, zeros, zeros)
    A_aileron = A_matrix(zeros, zeros, c, zeros)
    A_rolling = A_matrix(zeros, zeros, zeros, d)
    A_symmetric = A_matrix(a, b, zeros, zeros)
    A_asymmetric = A_matrix(zeros, zeros, c, d)
    A_total = A_matrix(a, b, c, d)
    planform_contribs = calculate_lift_drag_moment(A_planform)
    washout_contribs = calculate_lift_drag_moment(A_washout)
    aileron_contribs = calculate_lift_drag_moment(A_aileron)
    rolling_contribs = calculate_lift_drag_moment(A_rolling)
    symmetric_contribs = calculate_lift_drag_moment(A_symmetric)
    asymmetric_contribs = calculate_lift_drag_moment(A_asymmetric)
    total_contribs = calculate_lift_drag_moment(A_total)
    return planform_contribs, washout_contribs, aileron_contribs, rolling_contribs, symmetric_contribs, asymmetric_contribs, total_contribs

def calculate_lift_drag_moment(A):
    """Calculate lift, drag, rolling moment, and yawing moment coefficients."""
    CL = np.pi * Ra * A[0]
    CDi = np.pi * Ra * (-1/2 * A[1] * pbar + sum((j + 1) * A[j] ** 2 for j in range(N)))
    Cl = -np.pi * Ra / 4 * A[1]
    Cn = np.pi * Ra / 4 * (-1/2 * (A[0] + A[2]) * pbar + sum((2 * (j + 1) - 1) * A[j - 1] * A[j] for j in range(1, N)))
    return CL, CDi, Cl, Cn

def distribution_components(theta, a_coeff, b_coeff, c_coeff, d_coeff):
    """Calculate the lift distribution components for the planform, washout, aileron, and rolling."""
    CL_planform = 4 * bw / cw * aoa * np.array([sum(a_coeff[i] * np.sin((i + 1) * theta[j]) for i in range(N)) for j in range(N)])
    CL_washout = -4 * bw / cw * Omega * np.array([sum(b_coeff[i] * np.sin((i + 1) * theta[j]) for i in range(N)) for j in range(N)])
    CL_aileron = 4 * bw / cw * das * np.array([sum(c_coeff[i] * np.sin((i + 1) * theta[j]) for i in range(N)) for j in range(N)])
    CL_rolling = 4 * bw / cw * pbar * np.array([sum(d_coeff[i] * np.sin((i + 1) * theta[j]) for i in range(N)) for j in range(N)])
    CL_total = CL_planform + CL_washout + CL_aileron + CL_rolling
    CL_symmetric = CL_planform + CL_washout
    CL_assymetric = CL_aileron + CL_rolling
    return CL_planform, CL_washout, CL_aileron, CL_rolling, CL_symmetric, CL_assymetric, CL_total

def plot_lift_distribution(components, labels, contributions, title, decimal_points=14):
    """Plot lift distribution across the wing for different components with contributions above each subplot."""
    num_components = len(components)
    num_rows = (num_components + 1) // 2  # Round up to accommodate planform plot + components
    fig, axs = plt.subplots(num_rows, 2, figsize=(18, 2.5 * num_rows))
    fig.suptitle(title)

    # Plot the planform in the first subplot
    ax = axs[0, 0]
    plot_planform(ax, theta)  # Pass the axis to plot_planform to control where it plots
    ax.set_title("Planform Geometry with Aileron")
    ax.set_xlabel("Spanwise Coordinate (z/b)")
    ax.set_ylabel("Chord Length (c/b)")

    # Plot lift distribution for each component in the remaining subplots
    for i, (component, label) in enumerate(zip(components, labels)):
        row = (i + 1) // 2  # Start at the second row for components
        col = (i + 1) % 2
        ax = axs[row, col] if num_rows > 1 else axs[col]

        # Plotting the component
        ax.plot(z, component, label=label)
        ax.set_xlabel('Spanwise Coordinate (z/b)', fontsize=10)
        ax.legend()
        ax.invert_xaxis()
        ax.grid(linestyle=':', linewidth=0.5)

        # Adding the contribution above each subplot
        CL_contrib, CDi_contrib, Cl_contrib, Cn_contrib = contributions[i]
        ax.set_ylabel(f'Lift Distribution\n{label}', fontsize=10)
        format_str = f'{{:.{decimal_points}f}}'
        ax.set_title(f'{label} Contributions:\n'
                    f'CL: {format_str.format(CL_contrib)}, CDi: {format_str.format(CDi_contrib)}, '
                    f'Cl: {format_str.format(Cl_contrib)}, Cn: {format_str.format(Cn_contrib)}',
                    fontsize=10)

    # Remove the empty subplot if there is an odd number of components + planform
    if (num_components + 1) % 2 != 0:
        fig.delaxes(axs[-1, -1])
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the rect to leave space for the suptitle
    fig.subplots_adjust(top=0.92)  # Adjust the top to leave space for the suptitle
    plt.show()

def plot_planform(ax, theta):
    """Plot the planform geometry with leading and trailing edges, and visualize the ailerons on the given axis."""
    # Calculate chord along the span
    cb = chord(theta) / bw
    z = theta2s(theta) / 2  # Convert theta to spanwise coordinate z

    # Define leading and trailing edges
    leading_edge = 0.25 * cb
    trailing_edge = -0.75 * cb
    ax.plot(z, leading_edge, 'k-', label='Leading Edge', lw=0.5)
    ax.plot(z, trailing_edge, 'k-', label='Trailing Edge', lw=0.5)

    # Draw chord lines for visual effect
    for zi, le, te in zip(z, leading_edge, trailing_edge):
        ax.plot([zi, zi], [le, te], 'b--', lw=0.5)
    
    # Aileron positions
    zb1, zb2 = sar / 2, sat / 2
    cb1, cb2 = car / bw, cat / bw
    y1, y2 = cb1 * (-0.75 + fcf_ar), cb2 * (-0.75 + fcf_at)
    m = (y2 - y1) / (zb2 - zb1)
    b = y1 - m * zb1

    # Plot ailerons and vertical lines at aileron edges
    for sign in [1, -1]:  # Right and mirrored left side
        ax.plot([sign * zb1, sign * zb2], [y1, y2], 'red', linestyle='--', label='Aileron' if sign == 1 else None, lw=0.5)
        ax.plot([sign * zb1, sign * zb1], [y1, cb1 * -0.75], 'red', linestyle='--', linewidth=0.5)
        ax.plot([sign * zb2, sign * zb2], [y2, cb2 * -0.75], 'red', linestyle='--', linewidth=0.5)
    
    # Highlight trailing edge within the aileron region
    for i, zi in enumerate(z):
        if zb1 < abs(zi) < zb2:
            aileron_te = m * abs(zi) + b
            ax.plot([zi, zi], [trailing_edge[i], aileron_te], color='orange', linestyle='--', lw=0.5)
    
    # Plot settings
    ax.set_aspect('equal', adjustable='box')
    ax.grid(linestyle=':', linewidth=0.5)

def print_aerodynamic_coefficients(CL, CDi, Cl, Cn, decimals=16):
    """Print aerodynamic coefficients with a specified number of decimal points."""
    format_str = f'{{:.{decimals}f}}'  # Create a format string for the specified number of decimal points
    print("\nTotal Aerodynamic Coefficients:")
    print(f'CL:  {format_str.format(CL)}')
    print(f'CDi: {format_str.format(CDi)}')
    print(f'Cl:  {format_str.format(Cl)}')
    print(f'Cn:  {format_str.format(Cn)}\n')

# Initial calculations
theta, z = cosine_clustering()
a_coeff, b_coeff, c_coeff, d_coeff = solve_fourier_coefficients(theta)

# Find the total lift for each component as a single value
CL_planform, CL_washout, CL_aileron, CL_rolling, CL_symmetric, CL_assymetric, CL_total = distribution_components(theta, a_coeff, b_coeff, c_coeff, d_coeff)
planform_contribs, washout_contribs, aileron_contribs, rolling_contribs, symmetric_contribs, asymmetric_contribs, total_contribs = calculate_lift_contributions(a_coeff, b_coeff, c_coeff, d_coeff)
lift_components = [CL_total, CL_planform, CL_aileron, CL_washout, CL_rolling, CL_symmetric, CL_assymetric]
labels = ['Total', 'Planform', 'Aileron', 'Washout', 'Rolling', 'Symmetric', 'Assymetric']
contributions = [total_contribs, planform_contribs, aileron_contribs, washout_contribs, rolling_contribs, symmetric_contribs, asymmetric_contribs]
print_aerodynamic_coefficients(*total_contribs)
plot_lift_distribution(lift_components, labels, contributions, 'Lift Distribution Components')