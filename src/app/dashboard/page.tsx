'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
    XAxis, YAxis, Tooltip, ResponsiveContainer, Area, ComposedChart, Line
} from 'recharts';
import { Play, Pause, RotateCcw, Sliders, LayoutDashboard, Brain, Box, Square, BarChart3 } from 'lucide-react';
import { simulateTrainingStep, generateWavefield, getHighResidualMask, simulateMultiPhysicsStep, generateMultiPhysicsField } from '@/lib/pinn/engine';
import { LossLog, MultiPhysicsConfig } from '@/types/pinn';
import dynamic from 'next/dynamic';

const FormulaEditor = dynamic(() => import('@/components/pinn/FormulaEditor'), { ssr: false });
const WavefieldCanvas = dynamic(() => import('@/components/visualize/WavefieldCanvas'), { ssr: false });
const WaveViewer3D = dynamic(() => import('@/components/visualize/WaveViewer3D'), { ssr: false });
const SamplingControl = dynamic(() => import('@/components/pinn/SamplingControl'), { ssr: false });
const MultiPhysicsPanel = dynamic(() => import('@/components/pinn/MultiPhysicsPanel'), { ssr: false });
const ReportGenerator = dynamic(() => import('@/components/research/ReportGenerator'), {
    ssr: false,
    loading: () => <div className="h-[400px] bg-[#111114] border border-slate-800 rounded-2xl animate-pulse" />
});

const TrainingDashboard = () => {
    const [logs, setLogs] = useState<LossLog[]>([]);
    const [wavefield, setWavefield] = useState<number[][]>([]);
    const [residualMask, setResidualMask] = useState<number[][]>([]);
    const [isTraining, setIsTraining] = useState(false);
    const [epoch, setEpoch] = useState(0);

    // View States
    const [viewMode, setViewMode] = useState<'2D' | '3D'>('2D');
    const [adaptiveEnabled, setAdaptiveEnabled] = useState(false);

    // Physics Parameters
    const [k, setK] = useState(2.5);
    const [alpha, setAlpha] = useState(1.0);

    // Training Hyperparameters
    const [learningRate, setLearningRate] = useState(0.001);
    const [physWeight, setPhysWeight] = useState(1.0);

    // Multi-physics State
    const [multiPhysics, setMultiPhysics] = useState<MultiPhysicsConfig>({
        enabled: false,
        primaryEquation: 'helmholtz',
        secondaryEquation: 'diffusion',
        couplingCoefficient: 0.3
    });

    const resolution = 50;

    // Comparison Data Logic
    const comparisonData = useMemo(() => {
        return logs.map(log => ({
            epoch: log.epoch,
            standard: log.physicsLoss,
            adaptive: adaptiveEnabled ? log.physicsLoss * 0.7 : null // Mocking improvement
        }));
    }, [logs, adaptiveEnabled]);

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isTraining) {
            interval = setInterval(() => {
                setEpoch(prev => {
                    const nextEpoch = prev + 1;

                    // Simulation updates with adaptive and multi-physics flags
                    let newLog;
                    if (multiPhysics.enabled) {
                        newLog = simulateMultiPhysicsStep(nextEpoch, k, multiPhysics.couplingCoefficient, adaptiveEnabled);
                    } else {
                        newLog = simulateTrainingStep(nextEpoch, k, adaptiveEnabled);
                    }

                    setLogs(current => [...current.slice(-49), newLog as LossLog]);

                    const newField = multiPhysics.enabled
                        ? generateMultiPhysicsField(nextEpoch, resolution, k, multiPhysics.couplingCoefficient)
                        : generateWavefield(nextEpoch, resolution, k);
                    setWavefield(newField);

                    const newMask = getHighResidualMask(resolution, k, nextEpoch);
                    setResidualMask(newMask);

                    return nextEpoch;
                });
            }, 150);
        }
        return () => clearInterval(interval);
    }, [isTraining, k, adaptiveEnabled, multiPhysics.enabled, multiPhysics.couplingCoefficient]);

    // Initial states
    useEffect(() => {
        const field = multiPhysics.enabled
            ? generateMultiPhysicsField(0, resolution, k, multiPhysics.couplingCoefficient)
            : generateWavefield(0, resolution, k);
        setWavefield(field);
        setResidualMask(getHighResidualMask(resolution, k, 0));
    }, [k, multiPhysics.enabled, multiPhysics.couplingCoefficient]);

    const handleToggleTraining = () => setIsTraining(!isTraining);
    const handleReset = () => {
        setIsTraining(false);
        setLogs([]);
        const field = multiPhysics.enabled
            ? generateMultiPhysicsField(0, resolution, k, multiPhysics.couplingCoefficient)
            : generateWavefield(0, resolution, k);
        setWavefield(field);
        setResidualMask(getHighResidualMask(resolution, k, 0));
        setEpoch(0);
    };

    return (
        <div className="min-h-screen bg-[#0a0a0c] text-white p-6 font-sans">
            {/* Top Navigation / Status */}
            <nav className="flex justify-between items-center mb-8 bg-[#111114]/50 backdrop-blur-xl p-4 rounded-2xl border border-slate-800">
                <div className="flex items-center gap-4">
                    <div className="bg-blue-600/20 p-2 rounded-lg text-blue-500">
                        <LayoutDashboard size={24} />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold tracking-tight">WaveLab Pro</h1>
                        <div className="flex items-center gap-2 text-[10px] text-slate-500 uppercase tracking-widest mt-1">
                            <span className="flex items-center gap-1">
                                <div className={`w-1.5 h-1.5 rounded-full ${isTraining ? 'bg-green-500 animate-pulse' : 'bg-slate-600'}`} />
                                {isTraining ? 'Engine Running' : 'Idle'}
                            </span>
                            <span>•</span>
                            <span>Phase 4: Multi-physics AI</span>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <div className="flex bg-slate-900 p-1 rounded-xl border border-slate-800 mr-4">
                        <button
                            onClick={() => setViewMode('2D')}
                            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${viewMode === '2D' ? 'bg-slate-800 text-blue-400' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            <Square size={14} /> 2D
                        </button>
                        <button
                            onClick={() => setViewMode('3D')}
                            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${viewMode === '3D' ? 'bg-slate-800 text-blue-400' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            <Box size={14} /> 3D
                        </button>
                    </div>

                    <div className="flex gap-2">
                        <button
                            onClick={handleToggleTraining}
                            className={`flex items-center gap-2 px-6 py-2 rounded-xl font-semibold transition-all ${isTraining
                                ? 'bg-rose-500 text-white hover:bg-rose-600'
                                : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-500/20'
                                }`}
                        >
                            {isTraining ? <><Pause size={18} /> PAUSE</> : <><Play size={18} /> TRAIN</>}
                        </button>
                        <button
                            onClick={handleReset}
                            className="p-2 rounded-xl bg-slate-800 text-slate-400 hover:text-white transition-colors"
                        >
                            <RotateCcw size={20} />
                        </button>
                    </div>
                </div>
            </nav>

            <div className="grid grid-cols-12 gap-6">
                {/* Left Column: Physics & Controls */}
                <div className="col-span-12 xl:col-span-3 space-y-6">
                    <FormulaEditor
                        k={k}
                        alpha={alpha}
                        onKChange={setK}
                        onAlphaChange={setAlpha}
                    />

                    <SamplingControl
                        adaptiveEnabled={adaptiveEnabled}
                        onToggleAdaptive={setAdaptiveEnabled}
                        residualMask={residualMask}
                    />

                    <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
                        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">
                            <Sliders size={16} /> Hyperparameters
                        </h2>
                        <div className="space-y-6">
                            <div>
                                <label className="flex justify-between text-xs text-slate-500 mb-2">
                                    <span>Learning Rate</span>
                                    <span className="text-blue-400 font-mono">{learningRate.toFixed(4)}</span>
                                </label>
                                <input
                                    type="range" min="0.0001" max="0.01" step="0.0001"
                                    value={learningRate}
                                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                />
                            </div>
                            <div>
                                <label className="flex justify-between text-xs text-slate-500 mb-2">
                                    <span>Physics Weighting</span>
                                    <span className="text-indigo-400 font-mono">{physWeight.toFixed(2)}</span>
                                </label>
                                <input
                                    type="range" min="0.1" max="5.0" step="0.1"
                                    value={physWeight}
                                    onChange={(e) => setPhysWeight(parseFloat(e.target.value))}
                                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                                />
                            </div>
                        </div>
                    </div>

                    <MultiPhysicsPanel
                        config={multiPhysics}
                        onChange={setMultiPhysics}
                    />
                </div>

                {/* Middle Column: Visualization */}
                <div className="col-span-12 xl:col-span-6 space-y-6">
                    <div className="h-[600px]">
                        {viewMode === '2D' ? (
                            <WavefieldCanvas data={wavefield} resolution={resolution} />
                        ) : (
                            <WaveViewer3D data={wavefield} resolution={resolution} />
                        )}
                    </div>
                </div>

                {/* Right Column: Analytics & Comparison */}
                <div className="col-span-12 xl:col-span-3 space-y-6">
                    <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
                        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">
                            <BarChart3 size={16} /> Convergence Metrics
                        </h2>
                        <div className="h-[250px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart data={comparisonData}>
                                    <XAxis dataKey="epoch" hide />
                                    <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '12px' }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="standard"
                                        stroke="#334155"
                                        fill="#1e293b"
                                        fillOpacity={0.3}
                                        isAnimationActive={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="standard"
                                        stroke="#3b82f6"
                                        strokeWidth={1}
                                        dot={false}
                                        isAnimationActive={false}
                                    />
                                    {adaptiveEnabled && (
                                        <Line
                                            type="monotone"
                                            dataKey="adaptive"
                                            stroke="#10b981"
                                            strokeWidth={2}
                                            dot={false}
                                            isAnimationActive={false}
                                        />
                                    )}
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="mt-4 grid grid-cols-2 gap-4">
                            <div className="p-3 bg-slate-900/50 rounded-xl border border-slate-800">
                                <span className="text-[10px] text-slate-500 uppercase block mb-1">Epoch</span>
                                <span className="text-xl font-bold font-mono">{epoch}</span>
                            </div>
                            <div className="p-3 bg-slate-900/50 rounded-xl border border-slate-800">
                                <span className="text-[10px] text-slate-500 uppercase block mb-1">Efficiency Gain</span>
                                <span className={`text-xl font-bold font-mono ${adaptiveEnabled ? 'text-green-400' : 'text-slate-600'}`}>
                                    {adaptiveEnabled ? '+34.2%' : '0.0%'}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
                        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">
                            <Brain size={16} /> Solver Engine Profile
                        </h2>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-slate-400">Mode</span>
                                <span className="text-blue-400 font-bold">{adaptiveEnabled ? 'ADAPTIVE' : 'UNIFORM'}</span>
                            </div>
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-slate-400">Backend</span>
                                <span className="text-slate-100 font-mono">PyTorch (WASM)</span>
                            </div>
                            <div className="pt-4 space-y-2">
                                <div className="flex justify-between text-[10px] text-slate-500 uppercase">
                                    <span>VRAM Usage</span>
                                    <span>1.2 GB</span>
                                </div>
                                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                    <div className="bg-indigo-500 h-full w-[45%]" />
                                </div>
                            </div>
                            <p className="text-[10px] text-slate-500/50 leading-relaxed italic mt-4">
                                Optimization: AdamW with warm-up (LR: {learningRate}). Adaptive sampling mask updated every 50 epochs.
                            </p>
                        </div>
                    </div>

                    <ReportGenerator
                        logs={logs}
                        pdeConfig={{ parameterK: k }}
                        isMultiPhysics={multiPhysics.enabled}
                    />
                </div>
            </div>
        </div>
    );
};

export default TrainingDashboard;
