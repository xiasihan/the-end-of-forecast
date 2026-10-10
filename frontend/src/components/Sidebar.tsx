import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Plus, FileText, Trash2, Search } from 'lucide-react';
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

  const statusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-yellow';
      case 'completed': return 'text-green';
      case 'failed': return 'text-red';
      default: return 'text-dim';
    }
  };

  return (
    <>
      <aside className="w-60 bg-surface border-r border-border flex flex-col shrink-0 overflow-hidden">
        <div className="p-3 border-b border-border">
          <button
            onClick={() => setShowNew(true)}
            className="w-full flex items-center gap-2 px-3 py-2 bg-blue/10 border border-blue/20 rounded-md text-blue text-sm font-medium hover:bg-blue/20 transition-colors"
          >
            <Plus size={16} />
            {t('sidebar.newProject')}
          </button>
          <div className="relative mt-2">
            <Search size={14} className="absolute left-2.5 top-2 text-dim" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t('sidebar.filter')}
              className="w-full bg-bg border border-border rounded-md pl-8 pr-2 py-1.5 text-xs text-text placeholder-dim focus:outline-none focus:border-blue/50"
            />
          </div>
        </div>
        <div className="flex-1 overflow-y-auto">
          {filtered.map((p: Project) => (
            <div
              key={p.id}
              onClick={() => handleSelect(p)}
              className={`flex items-center gap-2.5 px-3 py-2.5 cursor-pointer border-l-2 hover:bg-white/[0.03] transition-colors group ${
                selectedProjectId === p.id ? 'border-blue bg-blue/[0.04]' : 'border-transparent'
              }`}
            >
              <FileText size={14} className={statusColor(p.status)} />
              <div className="flex-1 min-w-0">
                <div className="text-sm text-text truncate">{p.name}</div>
                <div className="text-[10px] text-dim">
                  {p.task_type || 'idle'} · {p.n_rows ? `${p.n_rows} rows` : ''}
                </div>
              </div>
              <button
                onClick={(e) => handleDelete(e, p.id)}
                className="opacity-0 group-hover:opacity-100 text-dim hover:text-red transition-all"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      </aside>
      {showNew && (
        <NewProjectDialog onClose={() => setShowNew(false)} onCreated={() => { setShowNew(false); refetch(); }} />
      )}
    </>
  );
}