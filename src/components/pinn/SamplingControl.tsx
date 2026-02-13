'use client';

import React from 'react';
import { Target, AlertCircle, TrendingUp, Info } from 'lucide-react';

interface SamplingControlProps {
    adaptiveEnabled: boolean;
    onToggleAdaptive: (val: boolean) => void;
    residualMask: number[][];
}

const SamplingControl: React.FC<SamplingControlProps> = ({
    adaptiveEnabled,
    onToggleAdaptive,
    residualMask
}) => {
    const activePoints = residualMask.flat().filter(v => v === 1).length;

    return (
        <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 uppercase tracking-wider">
                    <Target size={16} /> Adaptive Sampling
                </h2>
                <div className="relative group">
                    <Info size={16} className="text-slate-600 cursor-help" />
                    <div className="absolute right-0 top-6 w-48 p-2 bg-slate-900 border border-slate-800 rounded-lg text-[10px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none">
                        Focuses neural network training on high-residual (error) regions to accelerate convergence.
                    </div>
                </div>
            </div>

            <div className="flex items-center justify-between p-4 bg-slate-900/50 rounded-xl border border-slate-800">
                <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${adaptiveEnabled ? 'bg-green-500/10 text-green-500' : 'bg-slate-800 text-slate-500'}`}>
                        <TrendingUp size={20} />
                    </div>
                    <div>
                        <p className="text-xs font-bold">Residual-based Sampling</p>
                        <p className="text-[10px] text-slate-500">Accelerate ∇²E correction</p>
                    </div>
                </div>
                <button
                    onClick={() => onToggleAdaptive(!adaptiveEnabled)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${adaptiveEnabled ? 'bg-blue-600' : 'bg-slate-700'}`}
                >
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${adaptiveEnabled ? 'left-7' : 'left-1'}`} />
                </button>
            </div>

            <div className="space-y-4">
                <div className="flex justify-between text-[10px] text-slate-500 uppercase tracking-widest">
                    <span>Sampling Density Map</span>
                    <span className="text-blue-400 font-mono">{activePoints} Adaptive Points</span>
                </div>

                <div className="grid grid-cols-20 gap-px p-1 bg-slate-950 rounded-lg border border-slate-900 aspect-square overflow-hidden">
                    {residualMask.filter((_, idx) => idx % 2 === 0).map((row, i) => (
                        row.filter((_, idx) => idx % 2 === 0).map((val, j) => (
                            <div
                                key={`${i}-${j}`}
                                className={`w-full aspect-square rounded-[1px] transition-all duration-300 ${val === 1
                                        ? 'bg-blue-500/80 shadow-[0_0_4px_rgba(59,130,246,0.5)] scale-110'
                                        : 'bg-slate-900/50'
                                    }`}
                            />
                        ))
                    ))}
                </div>
            </div>

            {adaptiveEnabled && (
                <div className="flex items-start gap-2 p-3 bg-blue-500/5 border border-blue-500/20 rounded-xl text-[10px] text-blue-400 leading-relaxed">
                    <AlertCircle size={14} className="shrink-0 mt-0.5" />
                    Adaptive mode active. PINN logic now prioritizes high-residual voxels discovered in the last epoch.
                </div>
            )}
        </div>
    );
};

export default SamplingControl;
