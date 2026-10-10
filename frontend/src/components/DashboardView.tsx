import { useTranslation } from 'react-i18next';
import { useStore } from '../store';
import { Download, Copy, Check } from 'lucide-react';
import { useState } from 'react';

export default function DashboardView() {
  const { t } = useTranslation();
  const { holdoutScore, validationScore, bestModel, rounds, llmLog } = useStore();
  const [copied, setCopied] = useState(false);

  const curlCommand = `curl -X POST http://localhost:8000/api/projects/PROJECT_ID/predict \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"data": [{"ds": "2026-09-01 00:00:00", ...}]}'`;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-4 gap-4">
        <MetricCard title={t('dashboard.validation')} value={validationScore} subtitle={t('dashboard.validationSub')} color="blue" />
        <MetricCard title={t('dashboard.holdout')} value={holdoutScore} subtitle={t('dashboard.holdoutSub')} color="green" />
        <MetricCard title={t('dashboard.bestModel')} value={bestModel || '—'} subtitle={t('dashboard.bestModelSub')} color="purple" isString />
        <MetricCard title={t('dashboard.rounds')} value={rounds.length} subtitle={t('dashboard.roundsSub')} color="orange" isString />
      </div>
      <div className="bg-surface border border-border rounded-lg p-5">
        <h3 className="text-sm font-bold text-text mb-3">🔮 {t('dashboard.deploymentTitle')}</h3>
        <div className="bg-bg border border-border rounded-md p-4 font-mono text-xs text-text/80 relative">
          <pre className="whitespace-pre-wrap">{curlCommand}</pre>
          <button onClick={() => { navigator.clipboard.writeText(curlCommand); setCopied(true); setTimeout(() => setCopied(false), 2000); }} className="absolute top-3 right-3 text-dim hover:text-text transition-colors">
            {copied ? <Check size={16} className="text-green" /> : <Copy size={16} />}
          </button>
        </div>
        <div className="flex gap-3 mt-3">
          <button className="btn-primary text-xs flex items-center gap-1.5"><Download size={14} />{t('dashboard.download')}</button>
        </div>
      </div>
      {llmLog.length > 0 && (
        <div className="bg-surface border border-border rounded-lg p-5">
          <h3 className="text-sm font-bold text-text mb-3">🤖 {t('dashboard.llmDecisions')}</h3>
          <div className="space-y-3">
            {llmLog.map((entry, i) => (
              <div key={i} className="bg-bg border border-border rounded-md p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow/10 text-yellow font-medium">{entry.agent}</span>
                  <span className="text-xs text-text">{entry.summary}</span>
                </div>
                {entry.rationale && <div className="text-xs text-dim italic pl-1 border-l-2 border-yellow/20 ml-1">{entry.rationale}</div>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ title, value, subtitle, color, isString }: { title: string; value: string|number|null; subtitle: string; color: string; isString?: boolean }) {
  const tc = `text-${color}`;
  return (
    <div className={`bg-surface border border-${color}/20 rounded-lg p-4`}>
      <div className="text-[10px] text-dim uppercase tracking-wide">{title}</div>
      <div className={`text-2xl font-mono font-bold mt-1 ${isString ? 'text-sm' : tc}`}>{isString ? String(value||'?') : typeof value==='number'?value.toFixed(4):'—'}</div>
      <div className="text-[10px] text-dim mt-1">{subtitle}</div>
    </div>
  );
}