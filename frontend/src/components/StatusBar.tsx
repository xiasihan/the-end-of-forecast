import { useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useStore } from '../store';
import { Download, Circle } from 'lucide-react';

export default function StatusBar() {
  const { id } = useParams<{ id: string }>();
  const { t } = useTranslation();
  const { isRunning, holdoutScore, bestModel, errors } = useStore();

  return (
    <footer className="h-9 bg-white border-t border-border/50 flex items-center px-5 text-[11px] shrink-0 gap-3">
      <span className="font-mono text-textSecondary font-medium">{id || t('statusBar.noProject')}</span>

      <span className="w-px h-4 bg-border" />

      <div className="flex items-center gap-1.5">
        <Circle size={6} className={isRunning ? 'fill-amber text-amber' : 'fill-green text-green'} />
        <span className={`font-semibold ${isRunning ? 'text-amber' : 'text-green'}`}>
          {isRunning ? t('statusBar.running') : t('statusBar.idle')}
        </span>
      </div>

      {bestModel && (
        <>
          <span className="w-px h-4 bg-border" />
          <span className="text-textSecondary">
            {t('statusBar.best')}: <span className="text-accent font-semibold font-mono">{bestModel}</span>
          </span>
        </>
      )}

      {holdoutScore != null && (
        <>
          <span className="w-px h-4 bg-border" />
          <span className="text-textSecondary">
            {t('statusBar.holdout')}: <span className="text-green font-semibold font-mono">{Number(holdoutScore).toFixed(4)}</span>
          </span>
        </>
      )}

      {errors.length > 0 && (
        <>
          <span className="w-px h-4 bg-border" />
          <span className="text-red font-semibold">{errors.length} {t('statusBar.errors')}</span>
        </>
      )}

      <div className="flex-1" />

      <button className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-textSecondary hover:text-text hover:bg-gray-100 font-medium transition-all">
        <Download size={12} />
        {t('statusBar.export')}
      </button>
    </footer>
  );
}