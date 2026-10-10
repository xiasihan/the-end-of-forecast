import { create } from 'zustand';
import type { Project, PhaseData, RoundData, LLMLogEntry } from './types';

interface AppState {
  projects: Project[];
  setProjects: (p: Project[]) => void;
  selectedProjectId: string | null;
  selectProject: (id: string) => void;
  isRunning: boolean;
  setIsRunning: (v: boolean) => void;
  phases: PhaseData[];
  updatePhase: (phase: number, data: Partial<PhaseData>) => void;
  rounds: RoundData[];
  addRound: (r: RoundData) => void;
  llmLog: LLMLogEntry[];
  addLLMLog: (entry: LLMLogEntry) => void;
  holdoutScore: number | null;
  setHoldoutScore: (s: number | null) => void;
  validationScore: number | null;
  setValidationScore: (s: number | null) => void;
  bestModel: string | null;
  setBestModel: (m: string | null) => void;
  errors: string[];
  addError: (e: string) => void;
  resetRun: () => void;
}

export const useStore = create<AppState>((set) => ({
  projects: [],
  setProjects: (projects) => set({ projects }),
  selectedProjectId: null,
  selectProject: (id) => set({ selectedProjectId: id, isRunning: false, phases: makePhases(), rounds: [], llmLog: [], holdoutScore: null, validationScore: null, bestModel: null, errors: [] }),
  isRunning: false,
  setIsRunning: (isRunning) => set({ isRunning }),
  phases: makePhases(),
  updatePhase: (phase, data) => set((s: AppState) => ({ phases: s.phases.map((p) => (p.phase === phase ? { ...p, ...data } : p)) })),
  rounds: [],
  addRound: (r) => set((s: AppState) => ({ rounds: [...s.rounds, r] })),
  llmLog: [],
  addLLMLog: (entry) => set((s: AppState) => ({ llmLog: [...s.llmLog, entry] })),
  holdoutScore: null,
  setHoldoutScore: (s) => set({ holdoutScore: s }),
  validationScore: null,
  setValidationScore: (s) => set({ validationScore: s }),
  bestModel: null,
  setBestModel: (m) => set({ bestModel: m }),
  errors: [],
  addError: (e) => set((s: AppState) => ({ errors: [...s.errors, e] })),
  resetRun: () => set({ isRunning: false, phases: makePhases(), rounds: [], llmLog: [], holdoutScore: null, validationScore: null, bestModel: null, errors: [] }),
}));

function makePhases(): PhaseData[] {
  return Array.from({ length: 11 }, (_, i) => ({
    phase: i, name: `Phase ${i}`, status: 'pending' as const,
  }));
}