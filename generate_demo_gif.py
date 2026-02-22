import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# --- FDTD Solver Logic (Simplified/Robust for GIF) ---
def solve_wave_2d(grid_size=128, steps=200, c=1.0):
    dx = 1.0 / grid_size
    dt = 0.5 * dx / c
    u = np.zeros((steps, grid_size, grid_size))
    
    # Initial Condition: Gaussian pulse at center
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    u[0] = np.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2))
    u[1] = u[0].copy()
    
    # Simulation Loop
    for t in range(1, steps - 1):
        laplacian = (np.roll(u[t], 1, axis=0) + np.roll(u[t], -1, axis=0) +
                     np.roll(u[t], 1, axis=1) + np.roll(u[t], -1, axis=1) - 
                     4 * u[t]) / (dx**2)
        u[t+1] = 2 * u[t] - u[t-1] + (c * dt)**2 * laplacian
        
        # Dirichlet Boundaries
        u[t+1, 0, :] = 0
        u[t+1, -1, :] = 0
        u[t+1, :, 0] = 0
        u[t+1, :, -1] = 0
        
    return u

# --- Animation Logic ---
def generate_gif(filename='assets/demo.gif', steps=200, fps=20):
    print(f"Simulating {steps} frames...")
    u = solve_wave_2d(steps=steps)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(u[0], extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    plt.colorbar(im, label='Field Intensity')
    ax.set_title(f"PINN-WaveLab: 2D Wave Propagation (Step 0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    def update(frame):
        im.set_array(u[frame])
        ax.set_title(f"PINN-WaveLab: 2D Wave Propagation (Step {frame})")
        return [im]

    print(f"Creating animation (FPS={fps})...")
    ani = FuncAnimation(fig, update, frames=steps, interval=1000/fps, blit=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print(f"Saving to {filename}...")
    ani.save(filename, writer='pillow', fps=fps)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    generate_gif(filename='assets/demo.gif', steps=200, fps=20)
