export interface PDEConfig {
    id?: string;
    name: string;
    equationType: 'helmholtz' | 'wave' | 'poisson';
    parameterK: number; // Wave number k
    domainSize: {
        x: [number, number];
        y: [number, number];
        z?: [number, number];
    };
    resolution: number;
    metadata?: Record<string, unknown>;
    createdAt?: string;
}

export interface BoundaryCondition {
    id?: string;
    modelId: string;
    type: 'dirichlet' | 'neumann' | 'robin';
    edge: 'top' | 'bottom' | 'left' | 'right' | 'front' | 'back';
    value: number;
}

export interface LossLog {
    id: string;
    modelId: string;
    epoch: number;
    dataLoss: number;
    physicsLoss: number;
    boundaryLoss: number;
    totalLoss: number;
    // Compatibility with snake_case backends
    physics_loss?: number;
    total_loss?: number;
    data_loss?: number;
    weightData: number;
    weightPhysics: number;
    timestamp: string;
}

export interface PINNModel {
    id: string;
    name: string;
    config: PDEConfig;
    status: 'draft' | 'training' | 'completed' | 'failed';
    currentEpoch: number;
    bestLoss: number;
}

export interface ResearchReport {
    id: string;
    modelId: string;
    title: string;
    content: string; // Markdown format
    createdAt: string;
    metadata?: {
        epochCount: number;
        avgLoss: number;
        physicsType: string;
    };
}

export interface MultiPhysicsConfig {
    enabled: boolean;
    primaryEquation: 'helmholtz' | 'wave';
    secondaryEquation: 'diffusion' | 'poisson';
    couplingCoefficient: number;
}
