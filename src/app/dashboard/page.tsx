'use client';

import React, { useState, useEffect } from 'react';
import {
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import { Play, Pause, RotateCcw, Zap, Sliders, LayoutDashboard, Brain } from 'lucide-react';
import { simulateTrainingStep, generateWavefield } from '@/lib/pinn/engine';
import { LossLog } from '@/types/pinn';
import FormulaEditor from '@/components/pinn/FormulaEditor';
import WavefieldCanvas from '@/components/visualize/WavefieldCanvas';

const TrainingDashboard = () => {
    const [logs, setLogs] = useState<LossLog[]>([]);
    const [wavefield, setWavefield] = useState<number[][]>([]);
    const [isTraining, setIsTraining] = useState(false);
    const [epoch, setEpoch] = useState(0);

    // Physics Parameters
    const [k, setK] = useState(2.5);
    const [alpha, setAlpha] = useState(1.0);

    // Training Hyperparameters
    const [learningRate, setLearningRate] = useState(0.001);
    const [physWeight, setPhysWeight] = useState(1.0);

    const resolution = 50;

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isTraining) {
            interval = setInterval(() => {
                setEpoch(prev => {
                    const nextEpoch = prev + 1;

                    // Simulation updates
                    const newLog = simulateTrainingStep(nextEpoch, k);
                    setLogs(current => [...current.slice(-49), newLog as LossLog]);

                    const newField = generateWavefield(nextEpoch, resolution, k);
                    setWavefield(newField);

                    return nextEpoch;
                });
            }, 200);
        }
        return () => clearInterval(interval);
    }, [isTraining, k]);

    // Initial wavefield
    useEffect(() => {
        setWavefield(generateWavefield(0, resolution, k));
    }, [k]);

    const handleToggleTraining = () => setIsTraining(!isTraining);
    const handleReset = () => {
        setIsTraining(false);
        setLogs([]);
        setWavefield(generateWavefield(0, resolution, k));
        setEpoch(0);
    };

    return (
        <div className="min-h-screen bg-[#0a0a0c] text-white p-6 font-sans">
            {/* Top Navigation / Status */}
            <nav className="flex justify-between items-center mb-10 bg-[#111114]/50 backdrop-blur-xl p-4 rounded-2xl border border-slate-800">
                <div className="flex items-center gap-4">
                    <div className="bg-blue-600/20 p-2 rounded-lg text-blue-500">
                        <LayoutDashboard size={24} />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold tracking-tight">WaveLab Control Center</h1>
                        <div className="flex items-center gap-2 text-[10px] text-slate-500 uppercase tracking-widest mt-1">
                            <span className="flex items-center gap-1">
                                <div className={`w-1.5 h-1.5 rounded-full ${isTraining ? 'bg-green-500 animate-pulse' : 'bg-slate-600'}`} />
                                {isTraining ? 'System Active' : 'System Standby'}
                            </span>
                            <span>•</span>
                            <span>Engine v2.0-Alpha</span>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-6">
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
                {/* Left Column: Physics & Math */}
                <div className="col-span-12 xl:col-span-3 space-y-6">
                    <FormulaEditor
                        k={k}
                        alpha={alpha}
                        onKChange={setK}
                        onAlphaChange={setAlpha}
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
                                    <span>Physics Loss Weight</span>
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
                </div>

                {/* Middle Column: Visualization */}
                <div className="col-span-12 xl:col-span-6 space-y-6">
                    <div className="h-full">
                        <WavefieldCanvas data={wavefield} resolution={resolution} />
                    </div>
                </div>

                {/* Right Column: Analytics */}
                <div className="col-span-12 xl:col-span-3 space-y-6">
                    <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
                        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">
                            <Brain size={16} /> Convergence Metrics
                        </h2>
                        <div className="h-[250px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={logs}>
                                    <defs>
                                        <linearGradient id="colorPhys" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                    <XAxis dataKey="epoch" hide />
                                    <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '12px' }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="physicsLoss"
                                        stroke="#6366f1"
                                        fillOpacity={1}
                                        fill="url(#colorPhys)"
                                        isAnimationActive={false}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="dataLoss"
                                        stroke="#3b82f6"
                                        fill="transparent"
                                        strokeDasharray="4 4"
                                        isAnimationActive={false}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="mt-4 grid grid-cols-2 gap-4">
                            <div className="p-3 bg-slate-900/50 rounded-xl border border-slate-800">
                                <span className="text-[10px] text-slate-500 uppercase block mb-1">Epoch</span>
                                <span className="text-xl font-bold font-mono">{epoch}</span>
                            </div>
                            <div className="p-3 bg-slate-900/50 rounded-xl border border-slate-800">
                                <span className="text-[10px] text-slate-500 uppercase block mb-1">Residual</span>
                                <span className="text-xl font-bold font-mono text-indigo-400">
                                    {logs.length > 0 ? (logs[logs.length - 1].physicsLoss * physWeight).toFixed(4) : "0.0000"}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
                        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">
                            <Zap size={16} /> Engine Status
                        </h2>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-slate-400">Solver Mode</span>
                                <span className="text-blue-400 font-bold">Auto-PINN</span>
                            </div>
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-slate-400">Precision</span>
                                <span className="text-slate-100 font-mono">float32</span>
                            </div>
                            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-4">
                                <div
                                    className="bg-blue-500 h-full transition-all duration-300"
                                    style={{ width: `${Math.min(100, (epoch / 500) * 100)}%` }}
                                />
                            </div>
                            <p className="text-[10px] text-slate-500 text-center uppercase tracking-widest">Training Progress</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TrainingDashboard;
