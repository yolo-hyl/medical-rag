import { useState } from 'react';
import { Bot, Send, AlertCircle, Trash2, Clock } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiSearchAgent } from '../api';
import type { AgentResult } from '../types';
import DocumentDrawer from '../components/DocumentDrawer';

interface QueryHistory {
  id: string;
  query: string;
  result: AgentResult;
  timestamp: Date;
}

export default function SearchAgentPage() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<QueryHistory[]>([]);

  const handleQuery = async () => {
    if (!query.trim() || loading) return;
    const q = query.trim();
    setLoading(true);
    setError(null);

    try {
      const result = await apiSearchAgent(q);
      const entry: QueryHistory = {
        id: Date.now().toString(),
        query: q,
        result,
        timestamp: new Date(),
      };
      setHistory((prev) => [entry, ...prev]);
      setQuery('');
    } catch (e) {
      setError(e instanceof Error ? e.message : '检索 Agent 请求失败');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <Bot size={22} className="text-blue-600" />
          检索 Agent
        </h2>
        <p className="text-sm text-slate-500 mt-1">智能检索代理，自动分解问题并进行多路检索</p>
      </div>

      {/* Input card */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-4">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入需要检索的医疗问题..."
          rows={3}
          disabled={loading}
          className="w-full border border-slate-200 rounded-lg px-3 py-3 text-sm text-slate-800 placeholder-slate-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-50 scrollbar-thin"
        />
        <div className="flex items-center justify-between mt-3">
          <p className="text-xs text-slate-400">Ctrl+Enter 发送</p>
          <button
            onClick={handleQuery}
            disabled={loading || !query.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
          >
            <Send size={14} className={loading ? 'animate-pulse' : ''} />
            {loading ? '检索中...' : '发起检索'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
          <AlertCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-slate-700">检索历史</h3>
            <button
              onClick={() => setHistory([])}
              className="flex items-center gap-1 text-xs text-slate-400 hover:text-red-500 transition-colors"
            >
              <Trash2 size={12} />
              清空
            </button>
          </div>

          <div className="space-y-4">
            {history.map((entry) => (
              <div
                key={entry.id}
                className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden"
              >
                {/* Query */}
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-100">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-sm font-medium text-slate-800">{entry.query}</p>
                    <div className="flex items-center gap-1 text-xs text-slate-400 flex-shrink-0">
                      <Clock size={11} />
                      {entry.timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                </div>

                {/* Sub-queries */}
                {entry.result.sub_queries && entry.result.sub_queries.length > 0 && (
                  <div className="px-4 py-3 border-b border-slate-100">
                    <p className="text-xs font-medium text-slate-500 mb-2">分解子查询</p>
                    <div className="flex flex-wrap gap-1.5">
                      {entry.result.sub_queries.map((q, i) => (
                        <span
                          key={i}
                          className="text-xs bg-blue-50 text-blue-700 border border-blue-200 px-2 py-0.5 rounded"
                        >
                          {q}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Answer */}
                <div className="px-4 py-3 text-sm text-slate-700 leading-relaxed">
                  <div className="prose">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.result.answer}</ReactMarkdown>
                  </div>
                </div>

                {/* Sources */}
                {entry.result.documents && entry.result.documents.length > 0 && (
                  <div className="px-4 pb-3 border-t border-slate-100 pt-3">
                    <DocumentDrawer docs={entry.result.documents} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
