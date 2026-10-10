import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Upload, X } from 'lucide-react';
import { useStore } from '../store';

interface Props {
  onClose: () => void;
  onCreated: () => void;
}

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
    setLoading(true);
    setError('');
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
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed');
    } finally { setLoading(false); }
  };

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-surface border border-border rounded-xl w-[520px] max-h-[90vh] overflow-y-auto p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-text">{t('newProject.title')}</h2>
          <button onClick={onClose} className="text-dim hover:text-text"><X size={20} /></button>
        </div>
        <div className="space-y-4">
          <div>
            <label className="block text-xs text-dim mb-1">{t('newProject.projectName')}</label>
            <input value={name} onChange={(e) => setName(e.target.value)}
              className="w-full bg-bg border border-border rounded-md px-3 py-2 text-sm text-text placeholder-dim focus:outline-none focus:border-blue/50"
              placeholder={t('newProject.namePlaceholder')} />
          </div>
          <div>
            <label className="block text-xs text-dim mb-1">{t('newProject.predictionGoal')}</label>
            <textarea value={description} onChange={(e) => setDescription(e.target.value)}
              className="w-full bg-bg border border-border rounded-md px-3 py-2 text-sm text-text placeholder-dim h-20 focus:outline-none focus:border-blue/50 resize-none"
              placeholder={t('newProject.goalPlaceholder')} />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-xs text-dim mb-1">{t('newProject.targetColumn')}</label>
              <input value={targetCol} onChange={(e) => setTargetCol(e.target.value)}
                className="w-full bg-bg border border-border rounded-md px-2 py-1.5 text-sm text-text focus:outline-none focus:border-blue/50" />
            </div>
            <div>
              <label className="block text-xs text-dim mb-1">{t('newProject.timeColumn')}</label>
              <input value={timeCol} onChange={(e) => setTimeCol(e.target.value)}
                className="w-full bg-bg border border-border rounded-md px-2 py-1.5 text-sm text-text focus:outline-none focus:border-blue/50" />
            </div>
            <div>
              <label className="block text-xs text-dim mb-1">{t('newProject.idColumn')}</label>
              <input value={idCol} onChange={(e) => setIdCol(e.target.value)}
                className="w-full bg-bg border border-border rounded-md px-2 py-1.5 text-sm text-text focus:outline-none focus:border-blue/50" />
            </div>
          </div>
          <div onClick={() => fileRef.current?.click()}
            className="border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer hover:border-blue/40 transition-colors">
            <input ref={fileRef} type="file" accept=".csv,.parquet,.pqt,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] || null)} className="hidden" />
            {file ? (
              <div className="text-text">
                <Upload size={24} className="mx-auto mb-2 text-green" />
                <div className="text-sm font-medium">{file.name}</div>
                <div className="text-xs text-dim">{(file.size / 1024).toFixed(1)} {t('newProject.fileSize')}</div>
              </div>
            ) : (
              <div className="text-dim">
                <Upload size={24} className="mx-auto mb-2" />
                <div className="text-sm">{t('newProject.uploadPrompt')}</div>
              </div>
            )}
          </div>
          {error && <div className="text-red text-sm bg-red/10 border border-red/20 rounded-md px-3 py-2">{error}</div>}
          <div className="flex gap-3 pt-2">
            <button onClick={onClose} className="flex-1 btn-danger">{t('newProject.cancel')}</button>
            <button onClick={handleSubmit} disabled={loading} className="flex-1 btn-primary disabled:opacity-50">
              {loading ? t('newProject.creating') : t('newProject.createButton')}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}