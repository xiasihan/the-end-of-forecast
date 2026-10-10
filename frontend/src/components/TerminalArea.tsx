import { useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useStore } from '../store';
import PhaseCard from './PhaseCard';
import DashboardView from './DashboardView';

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
    <div className="space-y-1">
      {errors.length > 0 && (
        <div className="bg-red/10 border border-red/20 rounded-lg p-4 mb-4">
          <div className="text-red font-medium text-sm mb-1">{t('terminal.pipelineError')}</div>
          {errors.map((e, i) => (
            <div key={i} className="text-red/80 text-xs font-mono whitespace-pre-wrap">{e}</div>
          ))}
        </div>
      )}
      {isRunning && (
        <div className="flex items-center gap-2 text-yellow text-xs mb-3 animate-pulse">
          <div className="w-2 h-2 rounded-full bg-yellow" />
          {t('terminal.running')} — {runningPhase ? `${t(`phaseNames.${runningPhase.phase}`)}` : t('terminal.initializing')}
        </div>
      )}
      {phases.filter((p) => p.status !== 'pending').map((phase) => (
        <PhaseCard key={phase.phase} phase={phase} />
      ))}
      {phases.every((p) => p.status === 'pending') && (
        <div className="flex items-center justify-center h-64 text-dim text-sm">
          <div className="text-center">
            <div className="text-3xl mb-3">🚀</div>
            <p>{t('terminal.empty')}</p>
          </div>
        </div>
      )}
    </div>
  );
}