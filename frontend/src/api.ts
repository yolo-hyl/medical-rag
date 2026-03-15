import type {
  SSEEvent,
  HealthStatus,
  IngestResult,
  SearchResult,
  AskResult,
  ChatResult,
  EvalResult,
  AgentResult,
  AuthUser,
  SessionItem,
  MessageItem,
} from './types';

// Auth helpers
function authHeaders(token?: string): Record<string, string> {
  const h: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) h['X-Auth-Token'] = token;
  return h;
}

function getStoredToken(): string | undefined {
  try {
    const raw = localStorage.getItem('auth_user');
    if (!raw) return undefined;
    return (JSON.parse(raw) as AuthUser).token;
  } catch {
    return undefined;
  }
}

// SSE parsing helper
async function* parseSSE(response: Response): AsyncGenerator<SSEEvent> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const trimmed = line.slice(6).trim();
      if (!trimmed) continue;
      try {
        const evt: SSEEvent = JSON.parse(trimmed);
        yield evt;
        if (evt.type === 'done') return;
      } catch {
        // skip malformed lines
      }
    }
  }
}

// POST /api/auth/register
export async function apiRegister(
  phone: string,
  password: string,
  role: 'doctor' | 'patient' | 'admin'
): Promise<{ user_id: string }> {
  const res = await fetch('/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ phone, password, role }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? '注册失败');
  }
  return res.json();
}

// POST /api/auth/login
export async function apiLogin(
  phone: string,
  password: string
): Promise<{ user_id: string; token: string; role: string }> {
  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ phone, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? '登录失败');
  }
  return res.json();
}

// GET /api/history/{serviceType}
export async function apiGetHistory(
  serviceType: 'chat' | 'agent',
  token: string
): Promise<SessionItem[]> {
  const res = await fetch(`/api/history/${serviceType}`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`获取历史失败: ${res.status}`);
  return res.json();
}

// GET /api/history/{serviceType}/{sessionId}
export async function apiGetSessionMessages(
  serviceType: 'chat' | 'agent',
  sessionId: string,
  token: string
): Promise<MessageItem[]> {
  const res = await fetch(`/api/history/${serviceType}/${sessionId}`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`获取会话消息失败: ${res.status}`);
  return res.json();
}

// DELETE /api/history/{serviceType}/{sessionId}
export async function apiDeleteSession(
  serviceType: 'chat' | 'agent',
  sessionId: string,
  token: string
): Promise<{ success: boolean }> {
  const res = await fetch(`/api/history/${serviceType}/${sessionId}`, {
    method: 'DELETE',
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`删除会话失败: ${res.status}`);
  return res.json();
}

// GET /api/health
export async function apiHealth(): Promise<HealthStatus> {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error(`Health check failed: ${res.status} ${res.statusText}`);
  return res.json();
}

// POST /api/ingest
export async function apiIngest(
  records: unknown[],
  dropExisting = false
): Promise<IngestResult> {
  const res = await fetch('/api/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ records, drop_old: dropExisting }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Ingest failed: ${res.status} - ${err}`);
  }
  return res.json();
}

// POST /api/search
export async function apiSearch(
  query: string,
  limit = 5,
  filter?: string,
  hybrid = false
): Promise<SearchResult[]> {
  const body: Record<string, unknown> = { query, limit, use_hybrid: hybrid };
  if (filter) body.filter_expr = filter;

  const res = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Search failed: ${res.status} - ${err}`);
  }
  const data = await res.json();
  // Backend returns { query, documents, search_time } — unwrap documents array
  return Array.isArray(data) ? data : (data.documents ?? []);
}

// POST /api/ask
export async function apiAsk(
  question: string
): Promise<AskResult> {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Ask failed: ${res.status} - ${err}`);
  }
  return res.json();
}

// POST /api/chat (non-streaming)
export async function apiChat(
  message: string,
  sessionId?: string
): Promise<ChatResult> {
  const body: Record<string, unknown> = { question: message };
  if (sessionId) body.session_id = sessionId;

  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Chat failed: ${res.status} - ${err}`);
  }
  return res.json();
}

// POST /api/chat/stream (streaming)
export async function* apiChatStream(
  message: string,
  sessionId?: string
): AsyncGenerator<SSEEvent> {
  const body: Record<string, unknown> = { question: message };
  if (sessionId) body.session_id = sessionId;

  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: authHeaders(getStoredToken()),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Chat stream failed: ${res.status} - ${err}`);
  }
  yield* parseSSE(res);
}

// POST /api/eval
export async function apiEval(
  evalFile: string,
  sampleSize?: number,
  queryField?: string,
  referenceField?: string
): Promise<EvalResult> {
  const body: Record<string, unknown> = { eval_file: evalFile };
  if (sampleSize !== undefined) body.sample_size = sampleSize;
  if (queryField) body.query_field = queryField;
  if (referenceField) body.reference_field = referenceField;

  const res = await fetch('/api/eval', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Eval failed: ${res.status} - ${err}`);
  }
  return res.json();
}

// POST /api/search-agent
export async function apiSearchAgent(
  query: string
): Promise<AgentResult> {
  const res = await fetch('/api/search-agent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: query }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Search agent failed: ${res.status} - ${err}`);
  }
  return res.json();
}

// POST /api/agent (non-streaming)
export async function apiAgent(
  query: string,
  sessionId?: string
): Promise<AgentResult> {
  const body: Record<string, unknown> = { question: query };
  if (sessionId) body.session_id = sessionId;

  const res = await fetch('/api/agent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Agent failed: ${res.status} - ${err}`);
  }
  return res.json();
}

// POST /api/agent/stream (streaming)
export async function* apiAgentStream(
  query: string,
  sessionId?: string
): AsyncGenerator<SSEEvent> {
  const body: Record<string, unknown> = { question: query };
  if (sessionId) body.session_id = sessionId;

  const res = await fetch('/api/agent/stream', {
    method: 'POST',
    headers: authHeaders(getStoredToken()),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Agent stream failed: ${res.status} - ${err}`);
  }
  yield* parseSSE(res);
}
