import { useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useStore } from '../store';
import { Download } from 'lucide-react';

export default function StatusBar() {
  const { id } = useParams<{ id: string }>();
  const { t } = useTranslation();
  const { isRunning, holdoutScore, bestModel, errors } = useStore();

  return (
    <footer className="h-8 bg-surface border-t border-border flex items-center px-4 text-[11px] text-dim shrink-0">
      <span className="font-mono">{id || t('statusBar.noProject')}</span>
      <span className="mx-3 text-border">|</span>
      <span className={isRunning ? 'text-yellow' : 'text-green'}>
        {isRunning ? `● ${t('statusBar.running')}` : `○ ${t('statusBar.idle')}`}
      </span>
      {bestModel && <><span className="mx-3 text-border">|</span><span>{t('statusBar.best')}: <span className="text-purple font-mono">{bestModel}</span></span></>}
      {holdoutScore != null && <><span className="mx-3 text-border">|</span><span>{t('statusBar.holdout')}: <span className="text-green font-mono">{holdoutScore.toFixed(4)}</span></span></>}
      {errors.length > 0 && <><span className="mx-3 text-border">|</span><span className="text-red">{errors.length} {t('statusBar.errors')}</span></>}
      <div className="flex-1" />
      <button className="flex items-center gap-1 text-dim hover:text-text transition-colors"><Download size={12} />{t('statusBar.export')}</button>
    </footer>
  );
}