import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def solve_helmholtz_2d_fdm(k, domain_size, grid_points, boundary_conditions):
    """
    Solves the 2D Helmholtz equation using the Finite Difference Method.
    Equation: ∇²u + k²u = f(x, y)
    
    Args:
        k (float): Wavenumber.
        domain_size (tuple): ((x_min, x_max), (y_min, y_max)).
        grid_points (int): Number of points in each dimension (Nx = Ny = grid_points).
        boundary_conditions (dict): Defines the BC type and values. 
                                    Example: {'type': 'dirichlet', 'value': 0}
                                    Currently, only constant Dirichlet is supported.

    Returns:
        tuple: (xx, yy, u) where xx, yy are meshgrid coordinates and u is the solution.
    """
    x_min, x_max = domain_size[0]
    y_min, y_max = domain_size[1]
    N = grid_points

    # Create the grid
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy = np.meshgrid(x, y)

    # Source function f(x, y) - for this problem, we assume f=0
    f = np.zeros((N, N))

    # Assemble the matrix A for the linear system Au = b
    N2 = N * N
    
    # 5-point stencil for the Laplacian
    main_diag = np.ones(N2) * (-4 / dx**2 + k**2)
    off_diag_x = np.ones(N2 - 1) * (1 / dx**2)
    off_diag_y = np.ones(N2 - N) * (1 / dy**2)
    
    # Remove connections at the boundaries of the grid
    off_diag_x[N-1::N] = 0

    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    offsets = [0, -1, 1, -N, N]
    
    A = sp.diags(diagonals, offsets, shape=(N2, N2), format='csr')

    # Assemble the right-hand side vector b
    b = f.flatten()

    # Apply Dirichlet boundary conditions
    bc_value = boundary_conditions.get('value', 0)
    
    # Get indices of boundary points
    boundary_indices = np.zeros((N, N), dtype=bool)
    boundary_indices[0, :] = True
    boundary_indices[-1, :] = True
    boundary_indices[:, 0] = True
    boundary_indices[:, -1] = True
    boundary_flat_indices = np.where(boundary_indices.flatten())[0]

    # Modify matrix A and vector b for BCs
    for i in boundary_flat_indices:
        A[i, :] = 0
        A[i, i] = 1
        b[i] = bc_value

    # Solve the linear system
    try:
        u_flat = spsolve(A, b)
        u = u_flat.reshape((N, N))
    except Exception as e:
        print(f"Failed to solve the linear system: {e}")
        # Return a zero field on failure
        u = np.zeros((N, N))

    return xx, yy, u

if __name__ == '__main__':
    # --- Example Usage ---
    K_WAVENUMBER = 5.0
    DOMAIN = ((-1, 1), (-1, 1))
    GRID_N = 100
    BCS = {'type': 'dirichlet', 'value': 0}

    print("Solving 2D Helmholtz equation with FDM...")
    xx_sol, yy_sol, u_sol = solve_helmholtz_2d_fdm(
        k=K_WAVENUMBER,
        domain_size=DOMAIN,
        grid_points=GRID_N,
        boundary_conditions=BCS
    )
    print("Solver finished.")

    # --- Visualization ---
    plt.figure(figsize=(8, 6))
    plt.imshow(u_sol, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    plt.colorbar(label='u(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'FDM Solution of Helmholtz Equation (k={K_WAVENUMBER})')
    plt.show()
