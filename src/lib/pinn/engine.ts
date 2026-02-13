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
 * Mock training engine to simulate loss convergence for the UI.
 */
export const simulateTrainingStep = (epoch: number, k: number) => {
    const dataLoss = 0.5 * Math.exp(-epoch / 50) + 0.05 * Math.random() + (k * 0.001);
    const physicsLoss = 0.8 * Math.exp(-epoch / 80) + 0.08 * Math.random();
    const boundaryLoss = 0.3 * Math.exp(-epoch / 30) + 0.02 * Math.random();

    // Weighting simulation: dynamic adjustment
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
