import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Plus, Search, Trash2, Circle, Sparkles } from 'lucide-react';
import { useStore } from '../store';
import type { Project } from '../types';
import NewProjectDialog from './NewProjectDialog';

export default function Sidebar() {
  const { t } = useTranslation();
  const { projects, setProjects, selectedProjectId, selectProject, resetRun } = useStore();
  const navigate = useNavigate();
  const [showNew, setShowNew] = useState(false);
  const [search, setSearch] = useState('');

  const { refetch } = useQuery({
    queryKey: ['projects'],
    queryFn: async () => {
      const res = await fetch('/api/projects');
      const data = await res.json();
      setProjects(data.projects || []);
      return data;
    },
    refetchInterval: 5000,
  });

  const filtered = (projects || []).filter((p: Project) =>
    p.name.toLowerCase().includes(search.toLowerCase()),
  );

  const handleSelect = (p: Project) => {
    selectProject(p.id);
    resetRun();
    navigate(`/projects/${p.id}`);
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    await fetch(`/api/projects/${id}`, { method: 'DELETE' });
    refetch();
    if (selectedProjectId === id) navigate('/');
  };

  const statusData = (status: string) => {
    switch (status) {
      case 'running': return { color: 'bg-amber', text: 'text-amber' };
      case 'completed': return { color: 'bg-green', text: 'text-green' };
      case 'failed': return { color: 'bg-red', text: 'text-red' };
      default: return { color: 'bg-gray-300', text: 'text-dim' };
    }
  };

  return (
    <>
      <aside className="w-60 bg-white border-r border-border/60 flex flex-col shrink-0 overflow-hidden">
        <div className="p-4 border-b border-border/40">
          <button
            onClick={() => setShowNew(true)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-accent text-white rounded-xl text-sm font-semibold shadow-sm shadow-accent/20 hover:shadow-md hover:shadow-accent/30 transition-all active:scale-[0.98]"
          >
            <Plus size={18} />
            {t('sidebar.newProject')}
          </button>
          <div className="relative mt-3">
            <Search size={14} className="absolute left-3 top-2.5 text-dim" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t('sidebar.filter')}
              className="input-field pl-9 text-xs"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-2 py-1">
          {filtered.length === 0 && !search && (
            <div className="flex flex-col items-center justify-center px-4 py-12 text-center">
              <Sparkles size={24} className="text-dim/40 mb-3" />
              <p className="text-xs text-dim">{t('sidebar.noProjects')}</p>
            </div>
          )}
          {filtered.map((p: Project) => {
            const s = statusData(p.status);
            return (
              <div
                key={p.id}
                onClick={() => handleSelect(p)}
                className={`flex items-center gap-3 px-3 py-2.5 mb-0.5 rounded-xl cursor-pointer transition-all group text-sm
                  ${selectedProjectId === p.id ? 'bg-accent-light text-accent font-medium' : 'text-textSecondary hover:bg-gray-100'}`}
              >
                <Circle size={8} className={`shrink-0 ${s.color}`} fill="currentColor" />
                <div className="flex-1 min-w-0">
                  <div className="truncate font-medium">{p.name}</div>
                  <div className="text-[10px] text-dim mt-0.5">
                    {p.task_type || '—'} {p.n_rows ? `· ${p.n_rows} rows` : ''}
                  </div>
                </div>
                <button
                  onClick={(e) => handleDelete(e, p.id)}
                  className="opacity-0 group-hover:opacity-100 text-dim hover:text-red transition-all p-0.5"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            );
          })}
        </div>
      </aside>
      {showNew && (
        <NewProjectDialog onClose={() => setShowNew(false)} onCreated={() => { setShowNew(false); refetch(); }} />
      )}
    </>
  );
}