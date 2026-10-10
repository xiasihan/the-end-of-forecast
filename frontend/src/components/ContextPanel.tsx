import { useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Play, Circle } from 'lucide-react';
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
    <aside className="w-80 bg-surface border-l border-border flex flex-col shrink-0 overflow-hidden">
      <div className="p-3 border-b border-border">
        <button onClick={startPipeline} disabled={!id||running}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-green/20 border border-green/30 rounded-lg text-green font-medium text-sm hover:bg-green/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
          <Play size={16} className={running?'animate-pulse':''} />
          {running ? t('context.running') : t('context.runPipeline')}
        </button>
      </div>
      <div className="p-3 border-b border-border">
        <div className="text-[10px] text-dim uppercase tracking-wide mb-2">{t('context.phaseProgress')}</div>
        <div className="space-y-0.5">
          {phases.map((p) => (
            <div key={p.phase} className="flex items-center gap-2 text-xs py-0.5">
              <Circle size={8} className={`shrink-0 ${p.status==='completed'?'fill-green text-green':p.status==='running'?'fill-yellow text-yellow animate-pulse':p.status==='failed'?'fill-red text-red':'text-dim/30'}`} />
              <span className={`font-mono text-dim ${p.status!=='pending'?'text-text/80':''}`}>P{p.phase}</span>
              <span className={`truncate ${p.status!=='pending'?'text-text/70':'text-dim/50'}`}>{t(`phaseNames.${p.phase}`)}</span>
              {p.elapsedMs&&<span className="text-dim/50 font-mono ml-auto text-[10px]">{(p.elapsedMs/1000).toFixed(1)}s</span>}
            </div>
          ))}
        </div>
      </div>
      <div className="p-3 border-b border-border">
        <div className="text-[10px] text-dim uppercase tracking-wide mb-2">{t('context.liveMetrics')}</div>
        <div className="space-y-2">
          <TR label={t('context.validation')} value={validationScore} />
          <TR label={t('context.holdout')} value={holdoutScore} />
          <TR label={t('context.bestModel')} value={bestModel} />
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        <div className="text-[10px] text-dim uppercase tracking-wide mb-2">{t('context.llmDecisions')}</div>
        <div className="space-y-2">
          {llmLog.length===0&&<div className="text-xs text-dim/50 italic">{t('context.noDecisions')}</div>}
          {llmLog.map((e,i)=>(
            <div key={i} className="bg-bg border border-border rounded-md p-2">
              <div className="text-[10px] text-yellow font-medium mb-0.5">{e.agent}</div>
              <div className="text-xs text-text/70">{e.summary}</div>
              {e.rationale&&<div className="text-[11px] text-dim/60 italic mt-1 leading-relaxed">{e.rationale.slice(0,120)}</div>}
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

function TR({ label, value }: { label: string; value: string|number|null }) {
  return <div className="flex justify-between items-center"><span className="text-xs text-dim">{label}</span><span className="text-xs font-mono text-text">{typeof value==='number'?value.toFixed(4):value||'—'}</span></div>;
}