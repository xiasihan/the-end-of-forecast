import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Globe, Sparkles } from 'lucide-react';

export default function Header() {
  const { t, i18n } = useTranslation();

  const toggleLang = () => {
    i18n.changeLanguage(i18n.language === 'zh' ? 'en' : 'zh');
  };

  return (
    <header className="h-14 bg-white/80 backdrop-blur-xl border-b border-border/50 flex items-center px-6 shrink-0 z-10">
      <Link to="/" className="flex items-center gap-2.5 text-text hover:opacity-80 transition-opacity">
        <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
          <Sparkles size={16} className="text-white" />
        </div>
        <span className="font-bold text-[15px] tracking-tight">{t('header.title')}</span>
      </Link>
      <div className="flex-1" />

      <button
        onClick={toggleLang}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-textSecondary hover:text-text hover:bg-gray-100 text-xs font-medium transition-all"
        title={t('language')}
      >
        <Globe size={14} />
        <span>{i18n.language === 'zh' ? 'English' : '中文'}</span>
      </button>

      <a
        href="/api/docs"
        target="_blank"
        className="ml-2 px-3 py-1.5 text-textSecondary hover:text-text text-xs font-medium rounded-lg hover:bg-gray-100 transition-all"
      >
        {t('header.apiDocs')}
      </a>
    </header>
  );
}