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
        <main className="flex-1 overflow-y-auto px-6 py-4">
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
    <div className="flex items-center justify-center h-full text-dim">
      <div className="text-center max-w-lg">
        <div className="text-5xl mb-6">🔮</div>
        <h1 className="text-2xl font-bold text-text mb-3">{t('welcome.title')}</h1>
        <p className="text-sm mb-6">{t('welcome.subtitle')}</p>
        <p className="text-xs text-dim">
          {t('welcome.hint').replace('New Project', t('sidebar.newProject'))}
        </p>
      </div>
    </div>
  );
}