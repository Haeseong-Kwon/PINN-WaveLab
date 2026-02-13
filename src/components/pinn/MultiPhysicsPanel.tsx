'use client';

import React from 'react';
import { Layers, Zap, Info, ChevronRight } from 'lucide-react';
import { MultiPhysicsConfig } from '@/types/pinn';

interface MultiPhysicsPanelProps {
    config: MultiPhysicsConfig;
    onChange: (config: MultiPhysicsConfig) => void;
}

const MultiPhysicsPanel: React.FC<MultiPhysicsPanelProps> = ({ config, onChange }) => {
    return (
        <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 uppercase tracking-wider">
                    <Layers size={16} /> Multi-Physics Setup
                </h2>
                <div className={`px-2 py-0.5 rounded text-[10px] font-bold ${config.enabled ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30' : 'bg-slate-800 text-slate-500'}`}>
                    {config.enabled ? 'ACTIVE' : 'DISABLED'}
                </div>
            </div>

            <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-slate-900/50 rounded-xl border border-slate-800">
                    <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-lg ${config.enabled ? 'bg-indigo-500/10 text-indigo-500' : 'bg-slate-800 text-slate-500'}`}>
                            <Zap size={18} />
                        </div>
                        <div className="text-xs">
                            <p className="font-bold">Enable Coupling</p>
                            <p className="text-slate-500 text-[10px]">Combine multiple PDE solvers</p>
                        </div>
                    </div>
                    <button
                        onClick={() => onChange({ ...config, enabled: !config.enabled })}
                        className={`relative w-10 h-5 rounded-full transition-colors ${config.enabled ? 'bg-indigo-600' : 'bg-slate-700'}`}
                    >
                        <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-all ${config.enabled ? 'left-5.5' : 'left-0.5'}`} />
                    </button>
                </div>

                {config.enabled && (
                    <div className="space-y-4 animate-in fade-in slide-in-from-top-2 duration-300">
                        <div className="grid grid-cols-2 gap-3">
                            <div className="p-3 bg-slate-950 rounded-xl border border-slate-900">
                                <span className="text-[10px] text-slate-500 uppercase block mb-1">Primary PDE</span>
                                <span className="text-xs font-mono text-blue-400">Helmholtz</span>
                            </div>
                            <div className="flex items-center justify-center text-slate-700">
                                <ChevronRight size={16} />
                            </div>
                            <div className="p-3 bg-slate-950 rounded-xl border border-slate-900">
                                <span className="text-[10px] text-slate-500 uppercase block mb-1">Secondary PDE</span>
                                <select
                                    value={config.secondaryEquation}
                                    onChange={(e) => onChange({ ...config, secondaryEquation: e.target.value as 'diffusion' | 'poisson' })}
                                    className="bg-transparent text-xs font-mono text-emerald-400 outline-none w-full cursor-pointer"
                                >
                                    <option value="diffusion" className="bg-slate-900">Diffusion</option>
                                    <option value="poisson" className="bg-slate-900">Poisson</option>
                                </select>
                            </div>
                        </div>

                        <div>
                            <label className="flex justify-between text-xs text-slate-500 mb-2">
                                <span>Coupling Coefficient (λ)</span>
                                <span className="text-indigo-400 font-mono">{config.couplingCoefficient.toFixed(2)}</span>
                            </label>
                            <input
                                type="range" min="0" max="1" step="0.01"
                                value={config.couplingCoefficient}
                                onChange={(e) => onChange({ ...config, couplingCoefficient: parseFloat(e.target.value) })}
                                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                            />
                            <div className="flex justify-between mt-1 text-[8px] text-slate-600 uppercase tracking-tighter">
                                <span>Decoupled</span>
                                <span>Fully Coupled</span>
                            </div>
                        </div>

                        <div className="p-3 bg-indigo-500/5 border border-indigo-500/20 rounded-xl">
                            <div className="flex items-start gap-2">
                                <Info size={14} className="text-indigo-400 shrink-0 mt-0.5" />
                                <p className="text-[10px] text-indigo-300/80 leading-relaxed font-light">
                                    학습 손실 함수에 다중 물리 항이 추가되었습니다. 결합 계수가 높을수록 두 물리 법칙 간의 상호작용이 강해지며 연구 난이도가 상승합니다.
                                </p>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default MultiPhysicsPanel;
