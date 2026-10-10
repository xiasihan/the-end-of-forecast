import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Globe } from 'lucide-react';

export default function Header() {
  const { t, i18n } = useTranslation();

  const toggleLang = () => {
    i18n.changeLanguage(i18n.language === 'zh' ? 'en' : 'zh');
  };

  return (
    <header className="h-12 bg-surface border-b border-border flex items-center px-4 shrink-0 z-10">
      <Link to="/" className="flex items-center gap-2.5 text-text hover:text-purple transition-colors">
        <span className="text-lg">🔮</span>
        <span className="font-bold text-sm tracking-tight">{t('header.title')}</span>
      </Link>
      <div className="flex-1" />
      <button
        onClick={toggleLang}
        className="flex items-center gap-1 text-dim hover:text-text text-xs transition-colors mr-4"
        title={t('language')}
      >
        <Globe size={14} />
        {i18n.language === 'zh' ? 'EN' : '中文'}
      </button>
      <a
        href="/api/docs"
        target="_blank"
        className="text-dim hover:text-text text-xs transition-colors"
      >
        {t('header.apiDocs')}
      </a>
    </header>
  );
}