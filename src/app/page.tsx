'use client';

import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import { Play, Pause, RotateCcw, Activity, Zap, Settings } from 'lucide-react';
import { simulateTrainingStep } from '@/lib/pinn/engine';
import { LossLog } from '@/types/pinn';

const Dashboard = () => {
  const [logs, setLogs] = useState<LossLog[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [waveNumber, setWaveNumber] = useState(2.5);
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isTraining) {
      interval = setInterval(() => {
        setEpoch(prev => {
          const nextEpoch = prev + 1;
          const newLog = simulateTrainingStep(nextEpoch, waveNumber);
          setLogs(current => [...current.slice(-49), newLog as LossLog]);
          return nextEpoch;
        });
      }, 500);
    }
    return () => clearInterval(interval);
  }, [isTraining, waveNumber]);

  const handleToggleTraining = () => setIsTraining(!isTraining);
  const handleReset = () => {
    setIsTraining(false);
    setLogs([]);
    setEpoch(0);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-slate-100 p-6 font-sans">
      {/* Header */}
      <header className="flex justify-between items-center mb-10">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
            PINN WaveLab
          </h1>
          <p className="text-slate-500 text-sm mt-1">Physics-Informed Neural Network Monitoring System</p>
        </div>
        <div className="flex gap-4">
          <button
            onClick={handleToggleTraining}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-full transition-all ${isTraining
              ? 'bg-rose-500/10 text-rose-500 border border-rose-500/20 hover:bg-rose-500/20'
              : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-500/20'
              }`}
          >
            {isTraining ? <><Pause size={18} /> Stop Training</> : <><Play size={18} /> Start Training</>}
          </button>
          <button
            onClick={handleReset}
            className="p-2.5 rounded-full bg-slate-800 text-slate-400 hover:text-white transition-colors"
          >
            <RotateCcw size={20} />
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar Controls */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">
              <Settings size={16} /> Helmholtz Parameters
            </h2>
            <div className="space-y-6">
              <div>
                <label className="block text-xs text-slate-500 mb-2 uppercase tracking-wide">Wave Number (k)</label>
                <input
                  type="range" min="0.1" max="10" step="0.1"
                  value={waveNumber}
                  onChange={(e) => setWaveNumber(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <div className="flex justify-between mt-2 text-sm">
                  <span className="text-slate-500">0.1</span>
                  <span className="text-blue-400 font-mono">{waveNumber}</span>
                  <span className="text-slate-500">10.0</span>
                </div>
              </div>

              <div>
                <label className="block text-xs text-slate-500 mb-2 uppercase tracking-wide">Domain Logic</label>
                <div className="p-3 bg-slate-900/50 rounded-lg border border-slate-800 text-xs text-slate-400 space-y-2">
                  <div className="flex justify-between">
                    <span>Dimension</span>
                    <span className="text-slate-200">2D (X, Y)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Boundary</span>
                    <span className="text-slate-200">Dirichlet</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">
              <Zap size={16} /> Live Stats
            </h2>
            <div className="space-y-4">
              <StatItem label="Current Epoch" value={epoch.toString()} color="blue" />
              <StatItem
                label="Physics Loss"
                value={logs.length > 0 ? logs[logs.length - 1].physicsLoss.toFixed(6) : "0.000000"}
                color="indigo"
              />
              <StatItem
                label="Data Loss"
                value={logs.length > 0 ? logs[logs.length - 1].dataLoss.toFixed(6) : "0.000000"}
                color="blue"
              />
            </div>
          </div>
        </div>

        {/* Main Charts Area */}
        <div className="lg:col-span-3 space-y-6">
          <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl overflow-hidden relative">
            <div className="flex justify-between items-center mb-8">
              <h2 className="flex items-center gap-2 text-lg font-medium">
                <Activity size={20} className="text-blue-500" />
                Loss Convergence Profile
              </h2>
              <div className="flex gap-4 text-xs">
                <span className="flex items-center gap-1.5 text-blue-400"><div className="w-2 h-2 rounded-full bg-blue-500" /> Data Loss</span>
                <span className="flex items-center gap-1.5 text-indigo-400"><div className="w-2 h-2 rounded-full bg-indigo-500" /> Physics Loss</span>
              </div>
            </div>

            <div className="h-[350px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={logs}>
                  <defs>
                    <linearGradient id="colorData" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorPhysics" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis
                    dataKey="epoch"
                    stroke="#475569"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    stroke="#475569"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(val) => val.toFixed(2)}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '12px' }}
                    itemStyle={{ color: '#f1f5f9' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="dataLoss"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorData)"
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="physicsLoss"
                    stroke="#6366f1"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorPhysics)"
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
              <h3 className="text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">Weight Distribution</h3>
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={logs}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="epoch" hide />
                    <YAxis stroke="#475569" fontSize={12} tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Line
                      type="stepAfter"
                      dataKey="weightPhysics"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
              <h3 className="text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">Engine Accuracy</h3>
              <div className="flex flex-col items-center justify-center h-[200px] text-center">
                <div className="relative w-32 h-32 flex items-center justify-center">
                  <svg className="w-full h-full transform -rotate-90">
                    <circle
                      cx="64" cy="64" r="58"
                      className="stroke-slate-800" strokeWidth="12" fill="none"
                    />
                    <circle
                      cx="64" cy="64" r="58"
                      className="stroke-blue-500 transition-all duration-500"
                      strokeWidth="12" fill="none"
                      strokeDasharray="364.4"
                      strokeDashoffset={364.4 - (364.4 * 0.94)}
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-2xl font-bold">94.8%</span>
                    <span className="text-[10px] text-slate-500 uppercase tracking-tighter">MSE Confidence</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const StatItem = ({ label, value, color }: { label: string, value: string, color: string }) => (
  <div className="flex justify-between items-end p-3 bg-slate-900/40 rounded-xl border border-slate-800/50">
    <div>
      <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">{label}</p>
      <p className="text-xl font-mono font-bold text-slate-100">{value}</p>
    </div>
    <div className={`w-2 h-2 rounded-full bg-${color}-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]`} />
  </div>
);

export default Dashboard;
