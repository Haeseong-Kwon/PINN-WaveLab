'use client';

import React, { useRef, useEffect } from 'react';
import { Maximize2, Layers } from 'lucide-react';

interface WavefieldCanvasProps {
    data: number[][];
    resolution: number;
}

const WavefieldCanvas: React.FC<WavefieldCanvasProps> = ({ data, resolution }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || data.length === 0) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const cellSize = canvas.width / resolution;

        // Draw Heatmap
        data.forEach((row, i) => {
            row.forEach((val, j) => {
                // Normalize val to [0, 255] for visualization
                // Assuming val is [0, 1] for simplicity in mock
                const intensity = Math.floor(val * 255);
                // Using a "Viridis-like" blue-to-yellow map
                // R: intensity, G: intensity * 0.5 + 50, B: 255 - intensity
                ctx.fillStyle = `rgb(${intensity * 0.2}, ${intensity * 0.6 + 40}, ${255 - intensity * 0.5})`;
                ctx.fillRect(j * cellSize, i * cellSize, cellSize + 0.5, cellSize + 0.5);
            });
        });

        // Add subtle scanline effect
        ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        for (let i = 0; i < canvas.height; i += 4) {
            ctx.fillRect(0, i, canvas.width, 1);
        }
    }, [data, resolution]);

    return (
        <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col h-full">
            <div className="flex justify-between items-center mb-6">
                <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 uppercase tracking-wider">
                    <Layers size={16} /> Real-time Wavefield Predictor
                </h2>
                <div className="flex gap-2 text-slate-600">
                    <Maximize2 size={16} className="cursor-pointer hover:text-slate-400 transition-colors" />
                </div>
            </div>

            <div className="flex-1 relative group cursor-crosshair">
                <div className="absolute inset-0 bg-blue-500/5 rounded-xl filter blur-xl group-hover:bg-blue-500/10 transition-all duration-700" />
                <canvas
                    ref={canvasRef}
                    width={400}
                    height={400}
                    className="relative w-full aspect-square bg-slate-900 rounded-xl border border-slate-800 shadow-inner block overflow-hidden"
                />

                {/* Overlay Labels */}
                <div className="absolute bottom-4 left-4 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-[10px] text-slate-400 font-mono">
                    DOMAIN: [-1.0, 1.0]²
                </div>
                <div className="absolute top-4 right-4 bg-blue-600/20 backdrop-blur-md px-3 py-1.5 rounded-lg border border-blue-500/30 text-[10px] text-blue-400 font-mono font-bold">
                    INFERENCE OVERLAY
                </div>
            </div>

            <div className="mt-6 flex justify-between text-[10px] text-slate-500 font-mono uppercase tracking-widest">
                <span>-1.0</span>
                <div className="flex gap-4">
                    <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-blue-900" /> LOW</span>
                    <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-blue-400" /> MID</span>
                    <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-yellow-400" /> HIGH</span>
                </div>
                <span>+1.0</span>
            </div>
        </div>
    );
};

export default WavefieldCanvas;
