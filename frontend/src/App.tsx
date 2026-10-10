import { Routes, Route } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import TerminalArea from './components/TerminalArea';
import ContextPanel from './components/ContextPanel';
import StatusBar from './components/StatusBar';

export default function App() {
  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden">
      <Header />
      <div className="flex-1 flex overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto bg-bg px-8 py-6">
          <Routes>
            <Route path="/" element={<WelcomePage />} />
            <Route path="/projects/:id" element={<TerminalArea />} />
          </Routes>
        </main>
        <ContextPanel />
      </div>
      <StatusBar />
    </div>
  );
}

function WelcomePage() {
  const { t } = useTranslation();
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center max-w-lg">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-accent-light mb-8">
          <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
            <path d="M6 18L18 6l12 12-12 12L6 18z" stroke="#4f46e5" strokeWidth="2.5" strokeLinejoin="round"/>
            <circle cx="18" cy="18" r="5" fill="#4f46e5" opacity="0.3"/>
            <path d="M16 14l4 4-4 4" stroke="#4f46e5" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <h1 className="text-3xl font-bold text-text tracking-tight mb-3">{t('welcome.title')}</h1>
        <p className="text-textSecondary text-base leading-relaxed mb-2">{t('welcome.subtitle')}</p>
        <p className="text-dim text-sm">{t('welcome.hint').replace('New Project', t('sidebar.newProject'))}</p>
      </div>
    </div>
  );
}