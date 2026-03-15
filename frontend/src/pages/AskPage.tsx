import { useState } from 'react';
import { MessageSquare, Send, Clock, AlertCircle, Trash2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiAsk } from '../api';
import type { AskResult } from '../types';
import DocumentDrawer from '../components/DocumentDrawer';

interface QueryHistory {
  id: string;
  question: string;
  result: AskResult;
  timestamp: Date;
}

export default function AskPage() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<QueryHistory[]>([]);

  const handleAsk = async () => {
    if (!question.trim() || loading) return;
    const q = question.trim();
    setLoading(true);
    setError(null);

    try {
      const result = await apiAsk(q);
      const entry: QueryHistory = {
        id: Date.now().toString(),
        question: q,
        result,
        timestamp: new Date(),
      };
      setHistory((prev) => [entry, ...prev]);
      setQuestion('');
    } catch (e) {
      setError(e instanceof Error ? e.message : '问答失败');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <MessageSquare size={22} className="text-blue-600" />
          单轮问答
        </h2>
        <p className="text-sm text-slate-500 mt-1">向医疗知识库提问，获取基于文档的答案</p>
      </div>

      {/* Input card */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-4">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入您的医疗问题，例如：糖尿病的主要症状有哪些？"
          rows={3}
          disabled={loading}
          className="w-full border border-slate-200 rounded-lg px-3 py-3 text-sm text-slate-800 placeholder-slate-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-50 scrollbar-thin"
        />
        <div className="flex items-center justify-between mt-3">
          <p className="text-xs text-slate-400">Ctrl+Enter 发送</p>
          <button
            onClick={handleAsk}
            disabled={loading || !question.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
          >
            <Send size={14} />
            {loading ? '获取答案中...' : '提问'}
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
            <h3 className="text-sm font-semibold text-slate-700">历史问答</h3>
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
                {/* Question */}
                <div className="px-4 py-3 bg-slate-50 border-b border-slate-100">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-sm font-medium text-slate-800">{entry.question}</p>
                    <div className="flex items-center gap-1 text-xs text-slate-400 flex-shrink-0">
                      <Clock size={11} />
                      {entry.timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                </div>

                {/* Answer */}
                <div className="px-4 py-3 text-sm text-slate-700 leading-relaxed">
                  <div className="prose">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.result.answer}</ReactMarkdown>
                  </div>
                </div>

                {/* Timing stats */}
                {(entry.result.elapsed_time !== undefined || entry.result.retrieval_time !== undefined) && (
                  <div className="px-4 py-2 bg-slate-50 border-t border-slate-100 flex flex-wrap gap-3">
                    {entry.result.elapsed_time !== undefined && (
                      <span className="text-xs text-slate-500">
                        总耗时: <strong>{entry.result.elapsed_time.toFixed(2)}s</strong>
                      </span>
                    )}
                    {entry.result.retrieval_time !== undefined && (
                      <span className="text-xs text-slate-500">
                        检索: <strong>{entry.result.retrieval_time.toFixed(2)}s</strong>
                      </span>
                    )}
                    {entry.result.generation_time !== undefined && (
                      <span className="text-xs text-slate-500">
                        生成: <strong>{entry.result.generation_time.toFixed(2)}s</strong>
                      </span>
                    )}
                  </div>
                )}

                {/* Sources */}
                {(() => {
                  const docs = entry.result.sources ?? entry.result.documents ?? [];
                  return docs.length > 0 ? (
                    <div className="px-4 pb-3 border-t border-slate-100 pt-3">
                      <DocumentDrawer docs={docs} />
                    </div>
                  ) : null;
                })()}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
