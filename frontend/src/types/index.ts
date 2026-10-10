export interface Project {
  id: string;
  name: string;
  description: string;
  status: 'idle' | 'running' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  n_rows?: number;
  n_cols?: number;
  task_type?: string;
  latest_holdout_score?: number;
}

export interface Run {
  id: string;
  project_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  started_at?: string;
  completed_at?: string;
  elapsed_seconds?: number;
  summary?: string;
}

export interface PhaseEvent {
  phase: number;
  name: string;
  timestamp: number;
}

export interface PhaseCompleteEvent {
  phase: number;
  status: 'completed' | 'failed';
  data?: Record<string, unknown>;
  elapsed_ms: number;
  logs: string[];
  timestamp: number;
}

export interface RoundEvent {
  round_num: number;
  n_trials: number;
  timestamp: number;
}

export interface RoundCompleteEvent {
  round_num: number;
  status: string;
  best_model?: string;
  best_metric?: number;
  best_metric_name?: string;
  trial_summaries: TrialSummary[];
  diagnosis?: Record<string, unknown>;
  elapsed_ms: number;
  timestamp: number;
}

export interface TrialSummary {
  id: string;
  model: string;
  value: number;
  std: number;
}

export interface LLMDecisionEvent {
  agent: string;
  decision_summary: string;
  rationale: string;
  timestamp: number;
}

export interface CompleteEvent {
  run_id: string;
  summary: string;
  holdout_score?: number;
  validation_score?: number;
  best_model?: string;
  best_model_metric?: string;
  best_model_value?: number;
  timestamp: number;
}

export interface ErrorEvent {
  phase?: number;
  message: string;
  timestamp: number;
}

export type SSEEvent =
  | { event: 'phase_start'; data: string }
  | { event: 'phase_complete'; data: string }
  | { event: 'round_start'; data: string }
  | { event: 'round_complete'; data: string }
  | { event: 'llm_decision'; data: string }
  | { event: 'complete'; data: string }
  | { event: 'error'; data: string };

export interface PhaseData {
  phase: number;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startedAt?: number;
  completedAt?: number;
  elapsedMs?: number;
  data?: Record<string, unknown>;
  error?: string;
}

export interface RoundData {
  num: number;
  nTrials: number;
  bestModel?: string;
  bestMetric?: number;
  bestMetricName?: string;
  trials: TrialSummary[];
  diagnosis?: Record<string, unknown>;
}

export interface LLMLogEntry {
  agent: string;
  summary: string;
  rationale: string;
  timestamp: number;
}

export const PHASE_NAMES: Record<number, string> = {
  0: 'Requirements Gathering',
  1: 'Data Understanding',
  2: 'Data Preparation',
  3: 'LeakGuard',
  4: 'Baseline Establishment',
  5: 'Experiment Planning & Execution',
  6: 'Evaluation & Error Analysis',
  7: 'Iterative Refinement',
  8: 'Interpretation & Explainability',
  9: 'Pipeline Construction',
  10: 'Drift Detection & Monitoring',
};