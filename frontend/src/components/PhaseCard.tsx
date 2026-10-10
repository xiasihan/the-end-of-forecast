import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { ChevronDown, CheckCircle, XCircle, Loader } from 'lucide-react';
import type { PhaseData } from '../types';

interface Props { phase: PhaseData; }

export default function PhaseCard({ phase }: Props) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(phase.status === 'running');
  const prevRunning = useRef(false);

  useEffect(() => {
    if (phase.status === 'running') { setExpanded(true); prevRunning.current = true; }
  }, [phase.status]);

  const Icon = phase.status === 'completed' ? CheckCircle : phase.status === 'failed' ? XCircle : Loader;
  const iconColor = phase.status === 'completed' ? 'text-green' : phase.status === 'failed' ? 'text-red' : 'text-yellow';

  const renderData = () => {
    const d = phase.data;
    if (!d) return null;
    switch (phase.phase) {
      case 1:
        return (
          <div className="space-y-2">
            <div className="grid grid-cols-4 gap-3">
              <StatBox label="Task Type" value={String(d.task_type || '?')} />
              <StatBox label="Rows" value={String(d.n_rows || '?')} />
              <StatBox label="Columns" value={String(d.n_cols || '?')} />
              <StatBox label="Missing" value={`${(Number(d.missing_ratio || 0) * 100).toFixed(1)}%`} />
            </div>
            {d.quality_flags && (d.quality_flags as string[]).length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {(d.quality_flags as string[]).map((f, i) => (
                  <span key={i} className="text-[11px] px-2 py-0.5 rounded-full bg-yellow/10 text-yellow border border-yellow/20">{f}</span>
                ))}
              </div>
            )}
            {d.diagnostic && (
              <div className="bg-bg rounded-md p-3 border border-border">
                <div className="text-[10px] text-dim uppercase tracking-wide mb-1">Fingerprint Diagnostic</div>
                <div className="text-xs text-text/80 whitespace-pre-wrap font-mono leading-relaxed">{String(d.diagnostic)}</div>
              </div>
            )}
          </div>
        );
      case 4:
        return (
          <div className="space-y-2">
            <div className="bg-bg rounded-md border border-border overflow-hidden">
              <table className="w-full text-xs">
                <thead><tr className="border-b border-border text-dim text-left"><th className="px-3 py-1.5 font-medium">Model</th><th className="px-3 py-1.5 font-medium">Value</th><th className="px-3 py-1.5 font-medium">Status</th></tr></thead>
                <tbody>{(d.baselines as Array<{model:string;value:number}>||[]).map((b,i)=>(
                  <tr key={i} className="border-b border-border/50 last:border-0">
                    <td className="px-3 py-1.5 font-mono text-text">{b.model}</td>
                    <td className="px-3 py-1.5 font-mono text-blue">{b.value?.toFixed(4)||'nan'}</td>
                    <td className="px-3 py-1.5">{b.value&&!isNaN(b.value)?<span className="text-green">✅</span>:<span className="text-red">❌</span>}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
            <div className="text-xs text-green">⚡ Baseline floor: {typeof d.baseline_floor==='number'?d.baseline_floor.toFixed(4):'?'} — complex models must beat this</div>
          </div>
        );
      case 6:
        return (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-3">
              <StatBox label="Validation Score" value={typeof d.validation_score==='number'?d.validation_score.toFixed(4):'?'} />
              <StatBox label="Holdout Score" value={typeof d.holdout_score==='number'?d.holdout_score.toFixed(4):'?'} />
            </div>
            {typeof d.holdout_score==='number'&&<div className="text-xs text-green">✅ Unbiased holdout estimate on held-out data</div>}
          </div>
        );
      default:
        return d ? <pre className="text-[11px] text-dim/70 font-mono whitespace-pre-wrap max-h-48 overflow-y-auto bg-bg rounded-md p-2 border border-border">{JSON.stringify(d,null,1)}</pre> : null;
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={`phase-card ${phase.status}`}>
      <div className="card-header" onClick={() => setExpanded(!expanded)}>
        <Icon size={16} className={`${iconColor} ${phase.status==='running'?'animate-spin':''}`} />
        <span className="text-xs text-purple font-mono">P{phase.phase}</span>
        <span className="text-sm text-text font-medium flex-1">{t(`phaseNames.${phase.phase}`)}</span>
        {phase.elapsedMs && <span className="text-[10px] text-dim font-mono">{(phase.elapsedMs/1000).toFixed(1)}s</span>}
        <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${phase.status==='completed'?'bg-green/10 text-green':''} ${phase.status==='failed'?'bg-red/10 text-red':''} ${phase.status==='running'?'bg-yellow/10 text-yellow':''}`}>{phase.status}</span>
        <ChevronDown size={14} className={`text-dim transition-transform ${expanded?'rotate-180':''}`} />
      </div>
      <AnimatePresence>
        {expanded && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="px-4 pb-4 space-y-2">
              {renderData()}
              {phase.error && <div className="bg-red/10 border border-red/20 rounded-md p-3"><div className="text-red text-xs font-mono whitespace-pre-wrap">{phase.error}</div></div>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function StatBox({ label, value }: { label: string; value: string }) {
  return <div className="bg-bg border border-border rounded-md p-2.5"><div className="text-[10px] text-dim uppercase tracking-wide">{label}</div><div className="text-sm font-mono text-text mt-0.5">{value}</div></div>;
}