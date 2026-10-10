import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { ChevronDown, CheckCircle, XCircle, Loader, Zap } from 'lucide-react';
import type { PhaseData } from '../types';

interface Props { phase: PhaseData; }

export default function PhaseCard({ phase }: Props) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(phase.status === 'running');

  const Icon = phase.status === 'completed' ? CheckCircle : phase.status === 'failed' ? XCircle : Loader;
  const iconColor = phase.status === 'completed' ? 'text-green' : phase.status === 'failed' ? 'text-red' : 'text-amber';
  const borderStyle = phase.status === 'running' ? 'border-amber/30 bg-amber-bg/30' : phase.status === 'failed' ? 'border-red/20 bg-red-bg/30' : phase.status === 'completed' ? 'border-green/10' : '';

  const renderData = () => {
    const d = phase.data;
    if (!d) return null;
    switch (phase.phase) {
      case 1:
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-4 gap-3">
              <Stat label="Task Type" value={String(d.task_type || '—')} />
              <Stat label="Rows" value={String(d.n_rows || '—')} />
              <Stat label="Columns" value={String(d.n_cols || '—')} />
              <Stat label="Missing" value={`${(Number(d.missing_ratio || 0) * 100).toFixed(1)}%`} />
            </div>
            {d.quality_flags && (d.quality_flags as string[]).length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {(d.quality_flags as string[]).map((f, i) => (
                  <span key={i} className="badge badge-amber">{f}</span>
                ))}
              </div>
            )}
            {d.diagnostic && (
              <div className="bg-gray-50 rounded-xl p-4 border border-border/40">
                <div className="text-[10px] text-dim uppercase tracking-widest font-bold mb-2">Fingerprint Diagnostic</div>
                <div className="text-xs text-textSecondary whitespace-pre-wrap font-mono leading-relaxed">{String(d.diagnostic)}</div>
              </div>
            )}
          </div>
        );
      case 4:
        return (
          <div className="space-y-3">
            <div className="bg-gray-50 rounded-xl border border-border/30 overflow-hidden">
              <table className="w-full text-xs">
                <thead><tr className="bg-gray-100/50 text-textSecondary text-left"><th className="px-4 py-2 font-semibold">Model</th><th className="px-4 py-2 font-semibold">Score</th><th className="px-4 py-2 font-semibold w-16">Status</th></tr></thead>
                <tbody>{(d.baselines as Array<{model:string;value:number}>||[]).map((b,i)=>(
                  <tr key={i} className="border-t border-border/20 hover:bg-white/50 transition-colors">
                    <td className="px-4 py-2 font-mono font-medium text-text">{b.model}</td>
                    <td className="px-4 py-2 font-mono text-accent font-semibold">{b.value?.toFixed(4)||'—'}</td>
                    <td className="px-4 py-2">{b.value&&!isNaN(b.value)?<CheckCircle size={14} className="text-green"/>:<XCircle size={14} className="text-red"/>}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
            <div className="flex items-center gap-2 text-green font-medium text-xs">
              <Zap size={14} />
              Baseline floor: {typeof d.baseline_floor==='number'?d.baseline_floor.toFixed(4):'—'} — complex models must beat this
            </div>
          </div>
        );
      case 6:
        return (
          <div className="grid grid-cols-2 gap-4">
            <Stat label="Validation Score" value={typeof d.validation_score==='number'?d.validation_score.toFixed(4):'—'} />
            <Stat label="Holdout Score" value={typeof d.holdout_score==='number'?d.holdout_score.toFixed(4):'—'} highlight />
            {typeof d.holdout_score==='number'&&<div className="col-span-2 text-xs text-green font-medium">✅ Unbiased holdout estimate — never seen during experimentation</div>}
          </div>
        );
      default:
        return d ? <pre className="text-[11px] text-dim font-mono whitespace-pre-wrap max-h-48 overflow-y-auto bg-gray-50 rounded-xl p-3 border border-border/30">{JSON.stringify(d,null,1)}</pre> : null;
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={`phase-card ${borderStyle}`}>
      <div className="card-header rounded-t-xl" onClick={() => setExpanded(!expanded)}>
        <Icon size={18} className={`${iconColor} ${phase.status==='running'?'animate-spin':''}`} />
        <span className="badge badge-purple text-[10px] font-bold">P{phase.phase}</span>
        <span className="text-sm font-semibold text-text flex-1">{t(`phaseNames.${phase.phase}`)}</span>
        {phase.elapsedMs && <span className="text-xs text-dim font-mono bg-gray-100 px-2 py-0.5 rounded-lg">{(phase.elapsedMs/1000).toFixed(1)}s</span>}
        <span className={`badge ${phase.status==='completed'?'badge-green':phase.status==='failed'?'badge-red':phase.status==='running'?'badge-amber':'badge-blue'}`}>{phase.status}</span>
        <ChevronDown size={16} className={`text-dim transition-transform duration-200 ${expanded?'rotate-180':''}`} />
      </div>
      <AnimatePresence>
        {expanded && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="px-5 pb-5">
              {renderData()}
              {phase.error && <div className="bg-red-bg border border-red/10 rounded-xl p-4 mt-3"><div className="text-red text-xs font-mono whitespace-pre-wrap">{phase.error}</div></div>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`rounded-xl p-3.5 border ${highlight ? 'bg-green-bg/50 border-green/15' : 'bg-gray-50 border-border/20'}`}>
      <div className="metric-label">{label}</div>
      <div className={`text-sm font-mono font-bold mt-1 ${highlight ? 'text-green' : 'text-text'}`}>{value}</div>
    </div>
  );
}