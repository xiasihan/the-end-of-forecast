import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Upload, X, Sparkles } from 'lucide-react';
import { useStore } from '../store';

interface Props { onClose: () => void; onCreated: () => void; }

export default function NewProjectDialog({ onClose, onCreated }: Props) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { selectProject } = useStore();
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [targetCol, setTargetCol] = useState('y');
  const [timeCol, setTimeCol] = useState('ds');
  const [idCol, setIdCol] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    if (!name || !file) { setError(t('newProject.errorRequired')); return; }
    setLoading(true); setError('');
    try {
      const form = new FormData();
      form.append('name', name);
      form.append('description', description);
      form.append('target_col', targetCol);
      form.append('time_col', timeCol);
      form.append('id_col', idCol);
      form.append('budget', 'auto');
      form.append('file', file);
      const res = await fetch('/api/projects', { method: 'POST', body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      selectProject(data.id);
      onCreated();
      navigate(`/projects/${data.id}`);
    } catch (e: unknown) { setError(e instanceof Error ? e.message : 'Failed'); }
    finally { setLoading(false); }
  };

  return (
    <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-white rounded-2xl shadow-popup w-[540px] max-h-[90vh] overflow-y-auto p-8" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-accent/10 flex items-center justify-center">
              <Sparkles size={18} className="text-accent" />
            </div>
            <h2 className="text-xl font-bold text-text">{t('newProject.title')}</h2>
          </div>
          <button onClick={onClose} className="text-dim hover:text-text transition-colors p-1 rounded-lg hover:bg-gray-100"><X size={20} /></button>
        </div>

        <div className="space-y-5">
          <div>
            <label className="block text-xs font-semibold text-textSecondary mb-1.5">{t('newProject.projectName')}</label>
            <input value={name} onChange={(e) => setName(e.target.value)} className="input-field" placeholder={t('newProject.namePlaceholder')} />
          </div>

          <div>
            <label className="block text-xs font-semibold text-textSecondary mb-1.5">{t('newProject.predictionGoal')}</label>
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} className="input-field h-24 resize-none" placeholder={t('newProject.goalPlaceholder')} />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-xs font-semibold text-textSecondary mb-1.5">{t('newProject.targetColumn')}</label>
              <input value={targetCol} onChange={(e) => setTargetCol(e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-xs font-semibold text-textSecondary mb-1.5">{t('newProject.timeColumn')}</label>
              <input value={timeCol} onChange={(e) => setTimeCol(e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="block text-xs font-semibold text-textSecondary mb-1.5">{t('newProject.idColumn')}</label>
              <input value={idCol} onChange={(e) => setIdCol(e.target.value)} className="input-field" />
            </div>
          </div>

          <div onClick={() => fileRef.current?.click()}
            className="border-2 border-dashed border-border rounded-2xl p-10 text-center cursor-pointer hover:border-accent/40 hover:bg-accent-light/50 transition-all">
            <input ref={fileRef} type="file" accept=".csv,.parquet,.pqt,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] || null)} className="hidden" />
            {file ? (
              <div>
                <Upload size={28} className="mx-auto mb-3 text-green" />
                <div className="text-sm font-semibold text-text">{file.name}</div>
                <div className="text-xs text-dim mt-1">{(file.size / 1024).toFixed(1)} {t('newProject.fileSize')}</div>
              </div>
            ) : (
              <div className="text-dim">
                <Upload size={28} className="mx-auto mb-3" />
                <div className="text-sm font-medium">{t('newProject.uploadPrompt')}</div>
              </div>
            )}
          </div>

          {error && <div className="bg-red-bg border border-red/20 rounded-xl px-4 py-3 text-red text-sm font-medium">{error}</div>}

          <div className="flex gap-3 pt-1">
            <button onClick={onClose} className="flex-1 btn-danger">{t('newProject.cancel')}</button>
            <button onClick={handleSubmit} disabled={loading} className="flex-1 btn-primary">
              {loading ? t('newProject.creating') : t('newProject.createButton')}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}