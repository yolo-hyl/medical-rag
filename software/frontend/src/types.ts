export type UserRole = 'doctor' | 'patient' | 'admin';

export interface AuthUser {
  user_id: string;
  token: string;
  role: UserRole;
  phone: string;
}

export interface SessionItem {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface MessageItem {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface SourceDoc {
  source?: string;
  source_name?: string;
  distance?: number;
  summary?: string;
  content_preview: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  documents?: SourceDoc[];
  clarificationQuestions?: string[];
  subQueries?: string[];
  progressStage?: string;
  streamingStages?: string[];
  rewrittenQuery?: string;
  backgroundInfo?: string;
  isStreaming?: boolean;
  timestamp: Date;
}

export type NavItem = {
  id: string;
  label: string;
  icon: string;
};

export interface SSEEvent {
  type: 'progress' | 'documents' | 'answer' | 'done' | 'error' | 'clarification' | 'sub_queries' | 'token' | 'rewrite' | 'background';
  message?: string;
  data?: unknown;
  questions?: string[];
  queries?: string[];
}

export interface HealthStatus {
  status: string;
  services?: Record<string, string | boolean | number>;
  [key: string]: unknown;
}

export interface IngestResult {
  success: boolean;
  message?: string;
  count?: number;
  [key: string]: unknown;
}

export interface SearchResult {
  id?: string;
  score?: number;
  content?: string;
  metadata?: Record<string, unknown>;
  source?: string;
  source_name?: string;
  content_preview?: string;
  [key: string]: unknown;
}

export interface AskResult {
  answer: string;
  question?: string;
  sources?: SourceDoc[];
  documents?: SourceDoc[];
  elapsed_time?: number;
  retrieval_time?: number;
  generation_time?: number;
  search_time?: number;
  [key: string]: unknown;
}

export interface ChatResult {
  answer: string;
  session_id?: string;
  documents?: SourceDoc[];
  [key: string]: unknown;
}

export interface EvalResult {
  metrics?: Record<string, number>;
  results?: unknown[];
  summary?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface AgentResult {
  answer: string;
  needs_clarification?: boolean;
  clarification_questions?: string[];
  documents?: SourceDoc[];
  sub_queries?: string[];
  [key: string]: unknown;
}
