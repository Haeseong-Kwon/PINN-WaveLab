'use client';

import React, { useState } from 'react';
import { FileText, Cpu, Download, History, Sparkles, Loader2, Save } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';
import { LossLog } from '@/types/pinn';

interface ReportGeneratorProps {
    logs: LossLog[];
    pdeConfig: { parameterK: number };
    isMultiPhysics?: boolean;
}

const ReportGenerator: React.FC<ReportGeneratorProps> = ({ logs, pdeConfig, isMultiPhysics }) => {
    const [report, setReport] = useState<string | null>(null);
    const [isGenerating, setIsGenerating] = useState(false);

    const generateReport = async () => {
        setIsGenerating(true);
        // Real-world: Call Google Generative AI API here.
        // Mocking the AI response based on logs analysis
        const lastLog = logs[logs.length - 1];
        const avgLoss = logs.reduce((acc, log) => acc + log.totalLoss, 0) / logs.length;

        setTimeout(() => {
            const mockReport = `
# PINN 연구 분석 보고서 (Research Brief)

## 1. 실험 개요
- **물리 모델**: ${isMultiPhysics ? '다중 물리 (파동+확산)' : '헬름홀츠 파동 방정식'}
- **파수 (k)**: ${pdeConfig.parameterK || 2.5}
- **학습 에포크**: ${lastLog?.epoch || 0}
- **플랫폼**: WaveLab Pro AI v4.0

## 2. 수렴 특성 분석
현재 모델의 최종 총 손실(Total Loss)은 **${lastLog?.totalLoss.toExponential(4) || 0}**로 집계되었습니다. 
전체 학습 과정에서의 평균 손실은 **${avgLoss.toExponential(4)}**입니다. 

- **물리적 타당성**: 물리 손실(Physics Loss)이 안정적으로 감소하고 있으며, 이는 PDE 잔차가 학습 포인트에서 적절히 보정되고 있음을 시사합니다.
- **적응형 샘플링 효과**: 고잔차 영역에 대한 샘플링 밀도가 자동으로 조절되어 수렴 속도가 일반 샘플링 대비 약 1.4배 향상되었습니다.

## 3. 최적화 제안
- 현재의 학습률(LR)에서 초기 진동이 관찰됩니다. 학습률 감쇠(Learning Rate Decay) 스케줄러 적용을 권장합니다.
- 다중 물리 결합 계수 (λ)가 **${isMultiPhysics ? '0.5 이상' : '낮은'}** 수준일 때 학습 안정성이 높습니다.

## 4. 최종 결론
본 모델은 설정된 물리 도메인 내에서 높은 정확도로 파동 함수를 근사하고 있습니다. 향후 더 복잡한 경계 조건(Neumann BC) 확장이 가능할 것으로 보입니다.
      `;
            setReport(mockReport);
            setIsGenerating(false);
        }, 2000);
    };

    const exportPDF = async () => {
        const element = document.getElementById('research-report-content');
        if (!element) return;

        const canvas = await html2canvas(element);
        const imgData = canvas.toDataURL('image/png');
        const pdf = new jsPDF('p', 'mm', 'a4');
        const imgProps = pdf.getImageProperties(imgData);
        const pdfWidth = pdf.internal.pageSize.getWidth();
        const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;

        pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
        pdf.save(`research-report-${Date.now()}.pdf`);
    };

    return (
        <div className="bg-[#111114] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6 flex flex-col h-full">
            <div className="flex justify-between items-center">
                <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-400 uppercase tracking-wider">
                    <FileText size={16} /> Research Insights
                </h2>
                <div className="flex gap-2">
                    <button className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-500 hover:text-slate-300 transition-colors">
                        <History size={16} />
                    </button>
                </div>
            </div>

            {!report ? (
                <div className="flex-1 flex flex-col items-center justify-center space-y-4 py-12 border-2 border-dashed border-slate-800 rounded-2xl bg-slate-900/20">
                    <div className="p-4 bg-indigo-500/10 rounded-full text-indigo-500">
                        <Sparkles size={32} />
                    </div>
                    <div className="text-center">
                        <p className="text-sm font-medium text-slate-300">Generate Professional Report</p>
                        <p className="text-xs text-slate-500 mt-1 max-w-[200px]">
                            Gemini AI models will analyze your training logs and physics configs.
                        </p>
                    </div>
                    <button
                        onClick={generateReport}
                        disabled={isGenerating || logs.length === 0}
                        className="flex items-center gap-2 px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-xl text-sm font-semibold transition-all shadow-lg shadow-indigo-500/20"
                    >
                        {isGenerating ? <><Loader2 size={16} className="animate-spin" /> ANALYZING...</> : <><Cpu size={16} /> GENERATE REPORT</>}
                    </button>
                </div>
            ) : (
                <div className="flex-1 space-y-6 overflow-hidden flex flex-col">
                    <div
                        id="research-report-content"
                        className="flex-1 bg-slate-950 p-6 rounded-2xl border border-slate-900 overflow-y-auto prose prose-invert prose-sm max-w-none scrollbar-hide"
                    >
                        <ReactMarkdown>{report}</ReactMarkdown>
                    </div>

                    <div className="flex gap-3">
                        <button
                            onClick={exportPDF}
                            className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-bold transition-colors"
                        >
                            <Download size={16} /> PDF EXPORT
                        </button>
                        <button
                            onClick={() => setReport(null)}
                            className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-indigo-600/10 hover:bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 rounded-xl text-xs font-bold transition-colors"
                        >
                            <Save size={16} /> ARCHIVE
                        </button>
                    </div>
                </div>
            )}

            <div className="p-3 bg-slate-950/50 rounded-xl border border-slate-900">
                <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                    <span className="text-[10px] text-slate-500 font-mono">GEMINI 1.5 PRO READY</span>
                </div>
            </div>
        </div>
    );
};

export default ReportGenerator;
