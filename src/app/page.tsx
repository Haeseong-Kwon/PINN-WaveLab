'use client';

import React, { useState, useRef, useCallback } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { Play, Pause, Activity, Zap, Settings, Server, XCircle } from 'lucide-react';
import { LossLog } from '@/types/pinn';
import WavefieldCanvas from '@/components/visualize/WavefieldCanvas';
import { motion } from 'framer-motion';

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
    <div className="min-h-screen bg-[#020617] text-slate-100 p-8 font-sans selection:bg-blue-500/30 overflow-x-hidden">
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-between items-center mb-10 pb-6 border-b border-white/5 relative"
      >
        <div className="absolute top-0 left-0 w-full h-[500px] bg-blue-500/10 blur-[150px] pointer-events-none rounded-full" />
        <div className="relative z-10">
          <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-500 bg-clip-text text-transparent">
            PINN WaveLab <span className="text-2xl font-medium text-slate-500">v2.0</span>
          </h1>
          <p className="text-slate-400 text-sm mt-2 tracking-wide">Physics-Informed Neural Network Real-time Training Dashboard</p>
        </div>
        <div className="flex gap-4 relative z-10">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={isTraining ? handleStopTraining : handleStartTraining}
            className={`flex items-center gap-3 px-8 py-3 rounded-xl font-semibold transition-all shadow-xl backdrop-blur-md border ${isTraining
              ? 'bg-rose-500/10 text-rose-400 border-rose-500/20 hover:bg-rose-500/20 hover:border-rose-500/40 hover:shadow-rose-500/20'
              : 'bg-blue-600/90 text-white border-blue-500/50 hover:bg-blue-500 hover:border-blue-400 hover:shadow-blue-500/30'
              }`}
          >
            {isTraining ? <><Pause size={20} className="animate-pulse" /> Stop Simulation</> : <><Play size={20} /> Launch Neural Solver</>}
          </motion.button>
        </div>
      </motion.header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 relative z-10">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-1 space-y-8"
        >
          {/* Controls and Stats */}
          <ControlsPanel waveNumber={waveNumber} setWaveNumber={setWaveNumber} isTraining={isTraining} />
          <StatsPanel epoch={epoch} lastLog={lastLog} status={status} />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-3 space-y-8"
        >
          {/* Loss Chart */}
          <div className="bg-slate-900/50 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
            <LossChart logs={logs} />
          </div>

          {/* Wavefield Visualizers */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            <div className="bg-slate-900/40 backdrop-blur-lg border border-white/5 rounded-3xl p-6 shadow-xl hover:border-blue-500/30 transition-colors duration-500">
              <div className="h-[400px]">
                <WavefieldCanvas data={pinnPrediction} resolution={64} title="PINN Reconstructed Field" />
              </div>
            </div>
            <div className="bg-slate-900/40 backdrop-blur-lg border border-white/5 rounded-3xl p-6 shadow-xl hover:border-indigo-500/30 transition-colors duration-500">
              <div className="h-[400px]">
                <WavefieldCanvas data={fdmGroundTruth} resolution={64} title="FDM Ground Truth (Reference)" />
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

// --- Sub-components for better organization ---

const ControlsPanel = ({ waveNumber, setWaveNumber, isTraining }: { waveNumber: number, setWaveNumber: (val: number) => void, isTraining: boolean }) => (
  <div className="bg-slate-900/50 backdrop-blur-xl border border-white/10 rounded-3xl p-7 shadow-2xl relative overflow-hidden">
    <div className="absolute top-0 right-0 p-32 bg-blue-500/10 blur-[80px] rounded-full pointer-events-none" />
    <h2 className="flex items-center gap-3 text-sm font-bold text-slate-300 mb-8 uppercase tracking-widest relative z-10">
      <Settings size={18} className="text-blue-400" /> System Params
    </h2>
    <div className="space-y-6 relative z-10">
      <label className="block text-xs font-semibold text-slate-400 uppercase tracking-widest">Wave Number (k)</label>
      <input
        type="range" min="0.1" max="10" step="0.1"
        value={waveNumber}
        onChange={(e) => setWaveNumber(parseFloat(e.target.value))}
        disabled={isTraining}
        className="w-full h-2 bg-slate-800 rounded-full appearance-none cursor-pointer accent-blue-500 disabled:opacity-30 disabled:cursor-not-allowed hover:accent-blue-400 transition-all"
      />
      <div className="flex justify-between text-sm font-medium">
        <span className="text-slate-600">0.1</span>
        <motion.span
          key={waveNumber}
          initial={{ scale: 1.5, color: '#60a5fa' }}
          animate={{ scale: 1, color: '#38bdf8' }}
          className="font-mono bg-blue-500/10 px-3 py-1 rounded-lg border border-blue-500/20 shadow-inner"
        >
          {waveNumber.toFixed(1)}
        </motion.span>
        <span className="text-slate-600">10.0</span>
      </div>
    </div>
  </div>
);

const StatsPanel = ({ epoch, lastLog, status }: { epoch: number, lastLog: LossLog | null, status: string }) => {
  const getStatusIcon = () => {
    if (status.includes('Training') || status.includes('Solving')) return <Activity size={18} className="text-blue-400 animate-pulse" />;
    if (status.includes('Error') || status.includes('Divergence')) return <XCircle size={18} className="text-rose-500" />;
    return <Server size={18} className="text-slate-400" />;
  }

  return (
    <div className="bg-slate-900/50 backdrop-blur-xl border border-white/10 rounded-3xl p-7 shadow-2xl relative overflow-hidden">
      <div className="absolute bottom-0 left-0 p-32 bg-indigo-500/10 blur-[80px] rounded-full pointer-events-none" />
      <h2 className="flex items-center gap-3 text-sm font-bold text-slate-300 mb-8 uppercase tracking-widest relative z-10">
        <Zap size={18} className="text-indigo-400" /> Live Telemetry
      </h2>
      <div className="space-y-4 relative z-10">
        <StatItem label="Engine Status" value={status} icon={getStatusIcon()} highlight={status.includes('Training')} />
        <StatItem label="Training Epoch" value={epoch?.toString() ?? '0'} valueColor="text-blue-400" />
        <StatItem label="Physics Loss (PDE)" value={lastLog?.physics_loss ? lastLog.physics_loss.toExponential(4) : "0.00"} valueColor="text-indigo-400" />
        <StatItem label="Total Loss" value={lastLog?.total_loss ? lastLog.total_loss.toExponential(4) : "0.00"} valueColor="text-purple-400" />
      </div>
    </div>
  );
};

const StatItem = ({ label, value, icon, highlight, valueColor = "text-slate-100" }: { label: string, value: string, icon?: React.ReactNode, highlight?: boolean, valueColor?: string }) => (
  <motion.div
    layout
    className={`flex justify-between items-center p-4 rounded-2xl border transition-colors ${highlight ? 'bg-blue-500/10 border-blue-500/20 shadow-[0_0_15px_rgba(59,130,246,0.1)]' : 'bg-slate-800/40 border-white/5'}`}
  >
    <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</p>
    <div className="flex items-center gap-3">
      {icon}
      <p className={`text-sm font-mono font-semibold truncate ${valueColor}`}>{value}</p>
    </div>
  </motion.div>
);

const LossChart = ({ logs }: { logs: LossLog[] }) => (
  <>
    <div className="flex justify-between items-center mb-8 relative z-10">
      <h2 className="flex items-center gap-3 text-lg font-semibold tracking-wide text-slate-200">
        <Activity size={22} className="text-blue-500" />
        Convergence Trajectory
      </h2>
      {logs.length > 0 && (
        <span className="text-xs font-mono text-slate-500 bg-slate-800/80 px-3 py-1 rounded-full border border-slate-700">
          Epoch {logs[logs.length - 1].epoch}
        </span>
      )}
    </div>
    <div className="h-[380px] w-full relative z-10">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={logs} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.5} />
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorPhysics" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} opacity={0.5} />
          <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} tickMargin={10} />
          <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(val) => val.toExponential(1)} type="number" domain={['auto', 'auto']} tickMargin={10} />
          <Tooltip
            contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.5)', backdropFilter: 'blur(8px)' }}
            itemStyle={{ fontSize: '13px', fontWeight: '500', fontFamily: 'monospace' }}
            formatter={(value: unknown) => {
              if (typeof value === 'number') return value.toExponential(4);
              if (Array.isArray(value)) return value.map(v => typeof v === 'number' ? v.toExponential(4) : v).join(', ');
              return (value ?? '').toString();
            }}
          />
          <Area type="monotone" dataKey="total_loss" name="Total Loss" stroke="#8b5cf6" strokeWidth={3} fill="url(#colorTotal)" isAnimationActive={false} connectNulls />
          <Area type="monotone" dataKey="physics_loss" name="Physics Loss" stroke="#3b82f6" strokeWidth={3} fill="url(#colorPhysics)" isAnimationActive={false} connectNulls />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  </>
);

export default Dashboard;
