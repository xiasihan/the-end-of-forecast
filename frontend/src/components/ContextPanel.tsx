import { useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Play, Circle, Loader, Sparkles } from 'lucide-react';
import { useStore } from '../store';
import { useState } from 'react';

export default function ContextPanel() {
  const { id } = useParams<{ id: string }>();
  const { t } = useTranslation();
  const { phases, llmLog, isRunning, holdoutScore, validationScore, bestModel, setIsRunning, resetRun, addError, updatePhase, addRound, addLLMLog, setHoldoutScore, setValidationScore, setBestModel } = useStore();
  const [running, setRunning] = useState(false);

  const startPipeline = async () => {
    if (!id || running) return;
    setRunning(true); setIsRunning(true); resetRun();
    try {
      const res = await fetch(`/api/projects/${id}/runs?budget=auto`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      const { run_id } = await res.json();
      const es = new EventSource(`/api/projects/${id}/runs/${run_id}/stream`);
      es.addEventListener('phase_start', (e: MessageEvent) => { const d=JSON.parse(e.data); updatePhase(d.phase,{status:'running',startedAt:d.timestamp}); });
      es.addEventListener('phase_complete', (e: MessageEvent) => { const d=JSON.parse(e.data); updatePhase(d.phase,{status:d.status,data:d.data,elapsedMs:d.elapsed_ms,completedAt:d.timestamp}); });
      es.addEventListener('round_start', (e: MessageEvent) => { const d=JSON.parse(e.data); addRound({num:d.round_num,nTrials:d.n_trials,trials:[]}); });
      es.addEventListener('round_complete', (e: MessageEvent) => { const d=JSON.parse(e.data); addRound({num:d.round_num,nTrials:d.n_trials||0,bestModel:d.best_model,bestMetric:d.best_metric,bestMetricName:d.best_metric_name,trials:d.trial_summaries||[],diagnosis:d.diagnosis}); if(d.best_model)setBestModel(d.best_model); });
      es.addEventListener('llm_decision', (e: MessageEvent) => { const d=JSON.parse(e.data); addLLMLog({agent:d.agent,summary:d.decision_summary,rationale:d.rationale,timestamp:d.timestamp}); });
      es.addEventListener('complete', (e: MessageEvent) => { const d=JSON.parse(e.data); if(d.holdout_score!=null)setHoldoutScore(d.holdout_score); if(d.validation_score!=null)setValidationScore(d.validation_score); if(d.best_model)setBestModel(d.best_model); setIsRunning(false); setRunning(false); es.close(); });
      es.onerror = () => { setIsRunning(false); setRunning(false); es.close(); addError('SSE connection lost'); };
    } catch (err: unknown) { addError(err instanceof Error?err.message:'Failed'); setIsRunning(false); setRunning(false); }
  };

  return (
    <aside className="w-80 bg-white border-l border-border/60 flex flex-col shrink-0 overflow-hidden">
      {/* Run Button */}
      <div className="p-4 border-b border-border/40">
        <button onClick={startPipeline} disabled={!id||running}
          className="w-full flex items-center justify-center gap-2.5 px-5 py-3 bg-accent text-white rounded-2xl font-bold text-sm
                     shadow-lg shadow-accent/25 hover:shadow-xl hover:shadow-accent/30 hover:-translate-y-0.5
                     transition-all duration-200 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none">
          {running ? <Loader size={18} className="animate-spin" /> : <Play size={18} className="fill-white" />}
          {running ? t('context.running') : t('context.runPipeline')}
        </button>
      </div>

      {/* Phase Tracker */}
      <div className="px-4 py-3 border-b border-border/40">
        <div className="metric-label mb-3">{t('context.phaseProgress')}</div>
        <div className="space-y-0.5">
          {phases.map((p) => {
            const done = p.status === 'completed';
            const active = p.status === 'running';
            const failed = p.status === 'failed';
            return (
              <div key={p.phase} className={`flex items-center gap-2.5 text-xs py-1.5 px-2 rounded-lg transition-colors ${active ? 'bg-amber-bg' : done ? 'text-textSecondary' : 'text-dim/50'}`}>
                <Circle size={6} className={`shrink-0 ${done||active?'fill-current text-accent':failed?'fill-current text-red':'text-dim/30'}`} />
                <span className={`font-mono font-semibold ${done?'text-accent':active?'text-amber':''}`}>P{p.phase}</span>
                <span className={`truncate font-medium ${active?'text-amber':''}`}>{t(`phaseNames.${p.phase}`)}</span>
                {p.elapsedMs && <span className="text-dim font-mono ml-auto text-[10px]">{(p.elapsedMs/1000).toFixed(1)}s</span>}
              </div>
            );
          })}
        </div>
      </div>

      {/* Live Metrics */}
      <div className="px-4 py-3 border-b border-border/40">
        <div className="metric-label mb-3">{t('context.liveMetrics')}</div>
        <div className="space-y-2.5">
          <TR label={t('context.validation')} value={validationScore} color="text-blue" />
          <TR label={t('context.holdout')} value={holdoutScore} color="text-green" />
          <TR label={t('context.bestModel')} value={bestModel} color="text-purple" />
        </div>
      </div>

      {/* LLM Decisions */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="metric-label mb-3">{t('context.llmDecisions')}</div>
        <div className="space-y-2">
          {llmLog.length===0 && (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Sparkles size={20} className="text-dim/30 mb-3" />
              <p className="text-xs text-dim/60 italic">{t('context.noDecisions')}</p>
            </div>
          )}
          {llmLog.map((e,i)=>(
            <div key={i} className="bg-gray-50 border border-border/30 rounded-xl p-3 hover:border-accent/20 transition-all">
              <div className="flex items-center gap-1.5 mb-1.5">
                <span className="badge badge-purple text-[10px]">{e.agent}</span>
              </div>
              <div className="text-xs text-text font-medium mb-1">{e.summary}</div>
              {e.rationale && <div className="text-[11px] text-textSecondary/80 leading-relaxed">{e.rationale.slice(0,150)}</div>}
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

function TR({ label, value, color }: { label: string; value: string|number|null; color: string }) {
  return (
    <div className="flex justify-between items-center py-1">
      <span className="text-xs text-textSecondary font-medium">{label}</span>
      <span className={`text-xs font-mono font-bold ${color}`}>
        {typeof value==='number'?value.toFixed(4):value||'—'}
      </span>
    </div>
  );
}