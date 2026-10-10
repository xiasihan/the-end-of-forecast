import { useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useStore } from '../store';
import PhaseCard from './PhaseCard';
import DashboardView from './DashboardView';
import { Loader } from 'lucide-react';

export default function TerminalArea() {
  const { id } = useParams<{ id: string }>();
  const { phases, isRunning, errors } = useStore();
  const { t } = useTranslation();

  const runningPhase = phases.find((p) => p.status === 'running');
  const allDone = phases.every((p) => p.status === 'completed' || p.status === 'failed');

  if (allDone && !isRunning && phases.some((p) => p.status === 'completed')) {
    return <DashboardView />;
  }

  return (
    <div className="space-y-3">
      {errors.length > 0 && (
        <div className="bg-red-bg border border-red/20 rounded-2xl p-5 mb-2 animate-fade-in">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-red" />
            <div className="text-red font-semibold text-sm">{t('terminal.pipelineError')}</div>
          </div>
          {errors.map((e, i) => (
            <div key={i} className="text-red/80 text-xs font-mono bg-white/50 rounded-lg p-3 mt-1.5 whitespace-pre-wrap">{e}</div>
          ))}
        </div>
      )}

      {isRunning && (
        <div className="flex items-center gap-3 px-4 py-3 bg-amber-bg border border-amber/20 rounded-2xl animate-fade-in">
          <Loader size={16} className="text-amber animate-spin" />
          <span className="text-amber font-semibold text-sm">
            {t('terminal.running')}
          </span>
          {runningPhase && (
            <span className="text-amber/70 text-sm font-medium">
              — {t(`phaseNames.${runningPhase.phase}`)}
            </span>
          )}
        </div>
      )}

      {phases.filter((p) => p.status !== 'pending').map((phase, idx) => (
        <div key={phase.phase} className="animate-fade-in" style={{ animationDelay: `${idx * 60}ms` }}>
          <PhaseCard phase={phase} />
        </div>
      ))}

      {phases.every((p) => p.status === 'pending') && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent-light mb-4">
              <Loader size={28} className="text-accent/40" />
            </div>
            <p className="text-textSecondary text-sm font-medium">{t('terminal.empty')}</p>
          </div>
        </div>
      )}
    </div>
  );
}