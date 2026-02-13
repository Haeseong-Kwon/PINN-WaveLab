import { PDEConfig } from '@/types/pinn';

/**
 * Helmholtz Equation: ∇²E + k²E = 0
 * 
 * This engine handles the transformation of physical parameters into
 * tensor-ready formats and defines the physics loss function components.
 */

export class PINNPreprocessor {
    private config: PDEConfig;

    constructor(config: PDEConfig) {
        this.config = config;
    }

    /**
     * Normalizes domain coordinates to [-1, 1] for better neural network convergence.
     */
    normalizeCoordinates(x: number, y: number, z?: number) {
        const normX = 2 * (x - this.config.domainSize.x[0]) / (this.config.domainSize.x[1] - this.config.domainSize.x[0]) - 1;
        const normY = 2 * (y - this.config.domainSize.y[0]) / (this.config.domainSize.y[1] - this.config.domainSize.y[0]) - 1;

        if (z !== undefined && this.config.domainSize.z) {
            const normZ = 2 * (z - this.config.domainSize.z[0]) / (this.config.domainSize.z[1] - this.config.domainSize.z[0]) - 1;
            return { x: normX, y: normY, z: normZ };
        }

        return { x: normX, y: normY };
    }

    /**
     * Prepares the wave number k for scaling the physics loss.
     */
    getScaledK() {
        return this.config.parameterK;
    }

    /**
     * Generates a grid of collocation points for training.
     */
    generateCollocationPoints(numPoints: number) {
        const points = [];
        for (let i = 0; i < numPoints; i++) {
            points.push({
                x: Math.random() * (this.config.domainSize.x[1] - this.config.domainSize.x[0]) + this.config.domainSize.x[0],
                y: Math.random() * (this.config.domainSize.y[1] - this.config.domainSize.y[0]) + this.config.domainSize.y[0],
            });
        }
        return points;
    }
}

/**
 * Physics Loss Definition for Helmholtz Equation
 * In a real implementation with TensorFlow.js or ONNX, these would be gradient operations.
 */
export const computeHelmholtzResidual = (
    laplacianE: number, // ∇²E
    E: number,          // Field value
    k: number           // Wave number
) => {
    // Residue = ∇²E + k²E
    return laplacianE + Math.pow(k, 2) * E;
};

/**
 * Mock training engine with adaptive sampling capability.
 * When adaptive sampling is enabled, the convergence is accelerated in high-residual regions.
 */
export const simulateTrainingStep = (epoch: number, k: number, adaptiveEnabled: boolean = false) => {
    const speedFactor = adaptiveEnabled ? 1.5 : 1.0;

    const dataLoss = 0.5 * Math.exp(-(epoch * speedFactor) / 50) + 0.05 * Math.random() + (k * 0.001);
    const physicsLoss = 0.8 * Math.exp(-(epoch * speedFactor) / 80) + 0.08 * Math.random();
    const boundaryLoss = 0.3 * Math.exp(-(epoch * speedFactor) / 30) + 0.02 * Math.random();

    const weightPhysics = 1.0 + 0.5 * Math.sin(epoch / 10);
    const weightData = 1.0;

    return {
        epoch,
        dataLoss,
        physicsLoss,
        boundaryLoss,
        totalLoss: weightData * dataLoss + weightPhysics * physicsLoss + boundaryLoss,
        weightData,
        weightPhysics,
        timestamp: new Date().toISOString()
    };
};

/**
 * Identifies high-residual areas for adaptive sampling visualization.
 */
export const getHighResidualMask = (resolution: number, k: number, epoch: number) => {
    const mask = [];
    const noise = Math.max(0, 1 - epoch / 200);

    for (let i = 0; i < resolution; i++) {
        const row = [];
        for (let j = 0; j < resolution; j++) {
            const x = (j / resolution) * 2 - 1;
            const y = (i / resolution) * 2 - 1;
            // Residual peaks at boundaries and complex patterns
            const residual = Math.abs(Math.sin(k * x * 2) * Math.cos(k * y * 2)) * noise + (Math.random() * 0.1);
            row.push(residual > 0.4 ? 1 : 0);
        }
        mask.push(row);
    }
    return mask;
};

/**
 * Generates a 2D wavefield (heatmap data) based on the current epoch.
 * Simulates the field converging towards a solution.
 */
export const generateWavefield = (epoch: number, resolution: number, k: number) => {
    const grid = [];
    const convergence = 1 - Math.exp(-epoch / 100);

    for (let i = 0; i < resolution; i++) {
        const row = [];
        for (let j = 0; j < resolution; j++) {
            const x = (j / resolution) * 2 - 1;
            const y = (i / resolution) * 2 - 1;

            // Target pattern: sin(k*x) * cos(k*y)
            const target = (Math.sin(k * x * Math.PI) * Math.cos(k * y * Math.PI) + 1) / 2;

            // Random field for initial state
            const initial = Math.random();

            // Converged field
            const val = initial * (1 - convergence) + target * convergence;
            row.push(val);
        }
        grid.push(row);
    }
    return grid;
};

/**
 * Multi-physics simulation (Wave + Diffusion)
 * Simulates a system where wave propagation and diffusion coexist.
 */
export const simulateMultiPhysicsStep = (
    epoch: number,
    k: number,
    coupling: number = 0.5,
    adaptive: boolean = false
) => {
    const speedFactor = adaptive ? 1.5 : 1.0;
    const baseLoss = 0.5 * Math.exp(-(epoch * speedFactor) / 100);
    const physicsResidual = 0.4 * Math.exp(-(epoch * speedFactor) / 150) * (1 + 0.2 * Math.sin(epoch / 50));

    // Coupling effect: higher coupling slightly increases initial complexity but can stabilize later
    const couplingEffect = coupling * 0.1 * Math.cos(epoch / 100);
    const efficiencyGain = adaptive ? 0.7 : 1.0;

    const pLoss = (physicsResidual + couplingEffect) * efficiencyGain;
    const dLoss = baseLoss * efficiencyGain;

    return {
        epoch,
        dataLoss: dLoss,
        physicsLoss: pLoss,
        boundaryLoss: 0.05 * dLoss,
        totalLoss: dLoss + pLoss,
        weightData: 1.0,
        weightPhysics: 1.0 + coupling,
        timestamp: new Date().toISOString()
    };
};

/**
 * Generates a field representing multi-physics interaction.
 */
export const generateMultiPhysicsField = (epoch: number, resolution: number, k: number, coupling: number) => {
    const field: number[][] = [];
    const waveFreq = k;
    const diffusionRate = coupling * 0.2;
    const convergence = 1 - Math.exp(-epoch / 200);

    for (let i = 0; i < resolution; i++) {
        const row: number[] = [];
        for (let j = 0; j < resolution; j++) {
            const x = (j / resolution) * 2 - 1;
            const y = (i / resolution) * 2 - 1;
            const r = Math.sqrt(x * x + y * y);

            // Wave component (Target)
            const targetWave = (Math.sin(waveFreq * x * Math.PI) * Math.cos(waveFreq * y * Math.PI) + 1) / 2;

            // Diffusion component (Target)
            const targetDiffusion = Math.exp(-(r * r) / (0.1 + diffusionRate));

            // Combined Target
            const target = (1 - coupling) * targetWave + coupling * targetDiffusion;

            // Initial random state
            const initial = Math.random();

            // Converged field
            const val = initial * (1 - convergence) + target * convergence;
            row.push(val);
        }
        field.push(row);
    }
    return field;
};
