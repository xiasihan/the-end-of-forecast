import { useTranslation } from 'react-i18next';
import { useStore } from '../store';
import { Download, Copy, Check, Zap, Target, Gauge, Trophy, Layers } from 'lucide-react';
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
    <div className="space-y-6 animate-fade-in">
      {/* Metric Cards */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard icon={<Target size={20} />} title={t('dashboard.validation')} value={validationScore} subtitle={t('dashboard.validationSub')} accent="blue" />
        <MetricCard icon={<Zap size={20} />} title={t('dashboard.holdout')} value={holdoutScore} subtitle={t('dashboard.holdoutSub')} accent="green" />
        <MetricCard icon={<Trophy size={20} />} title={t('dashboard.bestModel')} value={bestModel} subtitle={t('dashboard.bestModelSub')} accent="purple" isString />
        <MetricCard icon={<Layers size={20} />} title={t('dashboard.rounds')} value={rounds.length} subtitle={t('dashboard.roundsSub')} accent="amber" isString />
      </div>

      {/* Deployment API */}
      <div className="card p-6">
        <h3 className="flex items-center gap-2 text-base font-bold text-text mb-4">
          <Gauge size={20} className="text-accent" />
          {t('dashboard.deploymentTitle')}
        </h3>
        <div className="bg-gray-50 rounded-xl p-4 font-mono text-xs text-textSecondary leading-relaxed relative group">
          <pre className="whitespace-pre-wrap">{curlCommand}</pre>
          <button
            onClick={() => { navigator.clipboard.writeText(curlCommand); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
            className="absolute top-3 right-3 p-2 rounded-lg bg-white border border-border text-dim hover:text-accent hover:border-accent/20 transition-all opacity-0 group-hover:opacity-100"
          >
            {copied ? <Check size={16} className="text-green" /> : <Copy size={16} />}
          </button>
        </div>
        <div className="flex gap-3 mt-4">
          <button className="btn-secondary text-xs flex items-center gap-1.5"><Download size={14} />{t('dashboard.download')}</button>
        </div>
      </div>

      {/* LLM Decision Log */}
      {llmLog.length > 0 && (
        <div className="card p-6">
          <h3 className="text-base font-bold text-text mb-4">🤖 {t('dashboard.llmDecisions')}</h3>
          <div className="space-y-3">
            {llmLog.map((entry, i) => (
              <div key={i} className="bg-gray-50/80 border border-border/30 rounded-xl p-4 hover:border-accent/20 transition-all">
                <div className="flex items-start gap-3">
                  <span className="badge badge-purple shrink-0 mt-0.5">{entry.agent}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold text-text mb-1">{entry.summary}</div>
                    {entry.rationale && (
                      <div className="text-xs text-textSecondary leading-relaxed border-l-2 border-accent/20 pl-3">{entry.rationale}</div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ icon, title, value, subtitle, accent, isString }: {
  icon: React.ReactNode; title: string; value: string | number | null; subtitle: string; accent: string; isString?: boolean;
}) {
  const accentMap: Record<string, string> = {
    blue: 'bg-blue-bg border-blue/15 text-blue',
    green: 'bg-green-bg border-green/15 text-green',
    purple: 'bg-purple-bg border-purple/15 text-purple',
    amber: 'bg-amber-bg border-amber/15 text-amber',
  };
  const display = isString ? String(value || '—') : typeof value === 'number' ? value.toFixed(4) : '—';

  return (
    <div className="card p-5 flex flex-col gap-3 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between">
        <span className="metric-label">{title}</span>
        <div className={`p-1.5 rounded-lg ${accentMap[accent]}`}>{icon}</div>
      </div>
      <div>
        <div className={`text-2xl font-bold tracking-tight ${isString ? 'text-sm text-accent' : 'font-mono text-text'}`}>{display}</div>
        <div className="text-[11px] text-dim mt-1">{subtitle}</div>
      </div>
    </div>
  );
}