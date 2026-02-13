import numpy as np
import matplotlib.pyplot as plt

def solve_wave_equation_2d_fdtd(
    domain_size=((-1, 1), (-1, 1)),
    grid_points=101,
    time_steps=200,
    c=1.0, # Wave speed
    initial_condition_type='gaussian'
):
    """
    Solves the 2D wave equation (u_tt = c^2 * (u_xx + u_yy)) using FDTD.

    Args:
        domain_size (tuple): ((x_min, x_max), (y_min, y_max)).
        grid_points (int): Number of points in each spatial dimension.
        time_steps (int): Number of time steps to simulate.
        c (float): Wave propagation speed.
        initial_condition_type (str): Type of initial pulse. 'gaussian' is supported.

    Returns:
        np.ndarray: A 3D array of shape (time_steps, grid_points, grid_points)
                    containing the wavefield at each time step.
    """
    x_min, x_max = domain_size[0]
    y_min, y_max = domain_size[1]
    N = grid_points
    
    # Grid setup
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Time setup (for stability, Courant-Friedrichs-Lewy condition)
    dt = (1 / c) * (1 / np.sqrt(1/dx**2 + 1/dy**2)) * 0.5 # 0.5 is a safety factor
    
    # Solution array
    u = np.zeros((time_steps, N, N))
    
    # --- Initial Condition ---
    if initial_condition_type == 'gaussian':
        # Gaussian pulse at the center
        xx, yy = np.meshgrid(x, y)
        sigma = 0.1
        u_initial = np.exp(-((xx**2 + yy**2) / (2 * sigma**2)))
        u[0] = u_initial
        u[1] = u_initial # Assume u_t(x, y, 0) = 0, so u at t=1 is same as t=0
    else:
        raise ValueError(f"Unknown initial condition type: {initial_condition_type}")

    # --- FDTD Simulation Loop ---
    # Pre-calculate constants
    cx2 = (c * dt / dx)**2
    cy2 = (c * dt / dy)**2

    for t in range(1, time_steps - 1):
        # Create padded arrays to handle boundaries easily
        u_padded = np.pad(u[t], pad_width=1, mode='constant', constant_values=0)
        
        # Central difference for Laplacian
        laplacian_u = (u_padded[1:-1, 2:] + u_padded[1:-1, :-2] - 2*u[t]) / dx**2 + \
                      (u_padded[2:, 1:-1] + u_padded[:-2, 1:-1] - 2*u[t]) / dy**2

        # Update rule from discretized wave equation
        u_next = 2*u[t] - u[t-1] + (c*dt)**2 * laplacian_u
        
        # Enforce Dirichlet boundary conditions (u=0 at boundaries)
        u_next[0, :] = 0
        u_next[-1, :] = 0
        u_next[:, 0] = 0
        u_next[:, -1] = 0
        
        u[t+1] = u_next

    return u

if __name__ == '__main__':
    print("Solving 2D Wave Equation with FDTD...")
    wavefield_3d = solve_wave_equation_2d_fdtd(grid_points=101, time_steps=300)
    print(f"Solver finished. Generated data shape: {wavefield_3d.shape}")

    # --- Visualization of a few frames ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    time_indices_to_plot = [10, 100, 250]
    for i, t_idx in enumerate(time_indices_to_plot):
        ax = axes[i]
        im = ax.imshow(wavefield_3d[t_idx], extent=[-1, 1, -1, 1], origin='lower', cmap='viridis', vmin=-0.5, vmax=1)
        ax.set_title(f'Wavefield at Time Step {t_idx}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax)
        
    plt.tight_layout()
    plt.show()
