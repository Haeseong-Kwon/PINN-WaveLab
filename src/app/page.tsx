'use client';

import React, { useState, useRef, useCallback } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { Play, Pause, Activity, Zap, Settings, Server, XCircle } from 'lucide-react';
import { LossLog } from '@/types/pinn';
import WavefieldCanvas from '@/components/visualize/WavefieldCanvas';

// Backend WebSocket URL
const WS_URL = 'ws://127.0.0.1:8000/ws/pinn-train';

const Dashboard = () => {
  const [logs, setLogs] = useState<LossLog[]>([]);
  const [status, setStatus] = useState('Idle');
  const [waveNumber, setWaveNumber] = useState(2.5);
  const [epoch, setEpoch] = useState(0);
  const [pinnPrediction, setPinnPrediction] = useState<number[][]>([]);
  const [fdmGroundTruth, setFdmGroundTruth] = useState<number[][]>([]);
  const ws = useRef<WebSocket | null>(null);

  const handleStartTraining = useCallback(() => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      console.log("WebSocket is already open.");
      return;
    }

    setLogs([]);
    setEpoch(0);
    setPinnPrediction([]);
    setFdmGroundTruth([]);
    setStatus('Connecting...');

    ws.current = new WebSocket(WS_URL);

    ws.current.onopen = () => {
      console.log('WebSocket connection established.');
      setStatus('Connected. Sending config...');
      const config = {
        k: waveNumber,
        epochs: 20000,
        lr: 0.001,
        resolution: 64,
      };
      ws.current?.send(JSON.stringify(config));
    };

    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.status === 'solving_fdm') {
        setStatus('Solving Ground Truth (FDM)...');
      } else if (message.status === 'fdm_solved') {
        setStatus('FDM solved. Starting PINN training...');
        setFdmGroundTruth(message.wavefield_ground_truth);
      } else if (message.status === 'starting_pinn_training') {
        setStatus('Training PINN...');
      } else if (message.status === 'diverged') {
        setStatus(`Divergence at epoch ${message.epoch}! Loss: ${message.loss.toFixed(2)}`);
        ws.current?.close();
      } else if (message.status === 'training_completed') {
        setStatus('Training Completed.');
        ws.current?.close();
      } else if (message.error) {
        setStatus(`Error: ${message.error}`);
        ws.current?.close();
      } else { // Regular epoch update
        setEpoch(message.epoch);
        setLogs(current => [...current.slice(-99), message as LossLog]);
        if (message.wavefield_prediction) {
          setPinnPrediction(message.wavefield_prediction);
        }
      }
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStatus('Connection Error. Is the Python server running?');
    };

    ws.current.onclose = () => {
      console.log('WebSocket connection closed.');
      if (status.startsWith('Training')) setStatus('Disconnected.');
      ws.current = null;
    };
  }, [waveNumber, status]);

  const handleStopTraining = () => {
    if (ws.current) {
      ws.current.close();
      setStatus('Manual Stop.');
    }
  };

  const isTraining = status.includes('Training') || status.includes('Solving');
  const lastLog = logs.length > 0 ? logs[logs.length - 1] : null;

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-slate-100 p-6 font-sans">
      <header className="flex justify-between items-center mb-10">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
            PINN WaveLab
          </h1>
          <p className="text-slate-500 text-sm mt-1">Real-time PINN Training & Visualization Backend</p>
        </div>
        <div className="flex gap-4">
          <button
            onClick={isTraining ? handleStopTraining : handleStartTraining}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-full transition-all ${isTraining
              ? 'bg-rose-500/10 text-rose-500 border border-rose-500/20 hover:bg-rose-500/20'
              : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-500/20'
              }`}
          >
            {isTraining ? <><Pause size={18} /> Stop</> : <><Play size={18} /> Start Training</>}
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-6">
          {/* Controls and Stats */}
          <ControlsPanel waveNumber={waveNumber} setWaveNumber={setWaveNumber} isTraining={isTraining} />
          <StatsPanel epoch={epoch} lastLog={lastLog} status={status} />
        </div>

        <div className="lg:col-span-3 space-y-6">
          {/* Loss Chart */}
          <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl overflow-hidden">
            <LossChart logs={logs} />
          </div>

          {/* Wavefield Visualizers */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <WavefieldCanvas data={pinnPrediction} resolution={64} title="PINN Prediction" />
            <WavefieldCanvas data={fdmGroundTruth} resolution={64} title="FDM Ground Truth" />
          </div>
        </div>
      </div>
    </div>
  );
};

// --- Sub-components for better organization ---

const ControlsPanel = ({ waveNumber, setWaveNumber, isTraining }: { waveNumber: number, setWaveNumber: (val: number) => void, isTraining: boolean }) => (
  <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
    <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">
      <Settings size={16} /> Parameters
    </h2>
    <div className="space-y-4">
      <label className="block text-xs text-slate-500 uppercase tracking-wide">Wave Number (k)</label>
      <input
        type="range" min="0.1" max="10" step="0.1"
        value={waveNumber}
        onChange={(e) => setWaveNumber(parseFloat(e.target.value))}
        disabled={isTraining}
        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500 disabled:opacity-50"
      />
      <div className="flex justify-between text-sm">
        <span className="text-slate-500">0.1</span>
        <span className="text-blue-400 font-mono">{waveNumber.toFixed(1)}</span>
        <span className="text-slate-500">10.0</span>
      </div>
    </div>
  </div>
);

const StatsPanel = ({ epoch, lastLog, status }: { epoch: number, lastLog: LossLog | null, status: string }) => {
  const getStatusIcon = () => {
    if (status.includes('Training') || status.includes('Solving')) return <Activity size={16} className="text-blue-400" />;
    if (status.includes('Error') || status.includes('Divergence')) return <XCircle size={16} className="text-rose-500" />;
    return <Server size={16} className="text-slate-500" />;
  }

  return (
    <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl">
      <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">
        <Zap size={16} /> Live Stats
      </h2>
      <div className="space-y-3">
        <StatItem label="Status" value={status} icon={getStatusIcon()} />
        <StatItem label="Current Epoch" value={epoch?.toString() ?? '0'} />
        <StatItem label="Physics Loss" value={lastLog?.physics_loss ? lastLog.physics_loss.toExponential(4) : "0.00"} />
        <StatItem label="Total Loss" value={lastLog?.total_loss ? lastLog.total_loss.toExponential(4) : "0.00"} />
      </div>
    </div>
  );
};

const StatItem = ({ label, value, icon }: { label: string, value: string, icon?: React.ReactNode }) => (
  <div className="flex justify-between items-center p-3 bg-slate-900/40 rounded-xl border border-slate-800/50">
    <p className="text-xs text-slate-400">{label}</p>
    <div className="flex items-center gap-2">
      {icon}
      <p className="text-sm font-mono font-medium text-slate-100 truncate">{value}</p>
    </div>
  </div>
);

const LossChart = ({ logs }: { logs: LossLog[] }) => (
  <>
    <h2 className="flex items-center gap-2 text-lg font-medium mb-8">
      <Activity size={20} className="text-blue-500" />
      Loss Convergence Profile
    </h2>
    <div className="h-[350px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={logs}>
          <defs>
            <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorPhysics" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis dataKey="epoch" stroke="#475569" fontSize={12} tickLine={false} axisLine={false} />
          <YAxis stroke="#475569" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => val.toExponential(1)} type="number" domain={['auto', 'auto']} />
          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
          <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '12px' }} formatter={(value: any) => {
            if (typeof value === 'number') return value.toExponential(4);
            return value;
          }} />
          <Area type="monotone" dataKey="total_loss" name="Total Loss" stroke="#3b82f6" strokeWidth={2} fill="url(#colorTotal)" isAnimationActive={false} connectNulls />
          <Area type="monotone" dataKey="physics_loss" name="Physics Loss" stroke="#6366f1" strokeWidth={2} fill="url(#colorPhysics)" isAnimationActive={false} connectNulls />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  </>
);

export default Dashboard;
