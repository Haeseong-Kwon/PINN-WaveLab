'use client';

import React, { useState, useEffect } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { Sigma, Info } from 'lucide-react';

interface FormulaEditorProps {
    k: number;
    alpha: number;
    onKChange: (val: number) => void;
    onAlphaChange: (val: number) => void;
}

const FormulaEditor: React.FC<FormulaEditorProps> = ({ k, alpha, onKChange, onAlphaChange }) => {
    const [latex, setLatex] = useState('');

    useEffect(() => {
        // Render Helmholtz Equation: ∇²E + k²E = 0
        // Simplified with alpha as Laplacian weight: α∇²E + k²E = 0
        const formula = `${alpha.toFixed(1)}\\nabla^2 E + ${k.toFixed(1)}^2 E = 0`;
        setLatex(formula);
    }, [k, alpha]);

    const containerRef = React.useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (containerRef.current) {
            katex.render(latex, containerRef.current, {
                throwOnError: false,
                displayMode: true,
            });
        }
    }, [latex]);

    return (
        <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 uppercase tracking-wider">
                    <Sigma size={16} /> PDE Formula Editor
                </h2>
                <Info size={16} className="text-slate-600 cursor-help" />
            </div>

            <div className="bg-slate-900/50 rounded-xl p-8 border border-slate-800/50 flex justify-center items-center min-h-[120px]">
                <div ref={containerRef} className="text-2xl text-blue-100" />
            </div>

            <div className="space-y-6">
                <div>
                    <label className="block text-xs text-slate-500 mb-3 uppercase tracking-wide">
                        Wave Number (k) - <span className="text-blue-400 font-mono">{k}</span>
                    </label>
                    <input
                        type="range" min="0.1" max="10" step="0.1"
                        value={k}
                        onChange={(e) => onKChange(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                </div>

                <div>
                    <label className="block text-xs text-slate-500 mb-3 uppercase tracking-wide">
                        Laplacian Weight (α) - <span className="text-indigo-400 font-mono">{alpha}</span>
                    </label>
                    <input
                        type="range" min="0.1" max="2" step="0.1"
                        value={alpha}
                        onChange={(e) => onAlphaChange(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                    />
                </div>
            </div>

            <div className="pt-4 border-t border-slate-800/50 text-[10px] text-slate-500 leading-relaxed italic">
                * Modifying coefficients dynamically updates the physics loss residue calculation in the PINN engine.
            </div>
        </div>
    );
};

export default FormulaEditor;
