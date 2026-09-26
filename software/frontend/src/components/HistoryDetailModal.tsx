import { useEffect, useState } from 'react';
import { X, User, Bot, Search, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiGetSessionMessages } from '../api';
import DocumentDrawer from './DocumentDrawer';
import type { SourceDoc } from '../types';

interface RichMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  extra_data?: string | null;
  // parsed from extra_data
  rewritten_query?: string;
  documents?: SourceDoc[];
  sub_queries?: string[];
}

interface HistoryDetailModalProps {
  sessionId: string;
  sessionTitle: string;
  serviceType: 'chat' | 'agent';
  token: string;
  onClose: () => void;
}

export default function HistoryDetailModal({
  sessionId,
  sessionTitle,
  serviceType,
  token,
  onClose,
}: HistoryDetailModalProps) {
  const [messages, setMessages] = useState<RichMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    apiGetSessionMessages(serviceType, sessionId, token)
      .then((raw) => {
        const parsed: RichMessage[] = raw.map((m) => {
          const msg: RichMessage = {
            role: m.role,
            content: m.content,
            timestamp: m.timestamp,
            extra_data: (m as Record<string, unknown>).extra_data as string | null,
          };
          if (msg.extra_data) {
            try {
              const extra = JSON.parse(msg.extra_data);
              msg.rewritten_query = extra.rewritten_query || '';
              msg.documents = extra.documents || [];
              msg.sub_queries = extra.sub_queries || [];
            } catch {
              // ignore parse error
            }
          }
          return msg;
        });
        setMessages(parsed);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [sessionId, serviceType, token]);

  // Pair user messages with the following assistant messages
  const pairs: Array<{ user: RichMessage; assistant: RichMessage | null }> = [];
  let i = 0;
  while (i < messages.length) {
    if (messages[i].role === 'user') {
      const user = messages[i];
      const assistant = messages[i + 1]?.role === 'assistant' ? messages[i + 1] : null;
      pairs.push({ user, assistant });
      i += assistant ? 2 : 1;
    } else {
      i++;
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 flex-shrink-0">
          <div>
            <h2 className="text-lg font-semibold text-slate-800">历史对话详情</h2>
            <p className="text-xs text-slate-400 mt-0.5 truncate max-w-sm">{sessionTitle || '（无标题）'}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6 scrollbar-thin">
          {loading && (
            <div className="flex justify-center py-12 text-slate-400 text-sm">加载中...</div>
          )}
          {error && (
            <div className="text-red-500 text-sm py-4">{error}</div>
          )}
          {!loading && !error && pairs.length === 0 && (
            <div className="text-slate-400 text-sm py-8 text-center">暂无对话记录</div>
          )}
          {pairs.map(({ user, assistant }, idx) => (
            <div key={idx} className="space-y-3">
              {/* Round label */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400 font-medium">第 {idx + 1} 轮</span>
                <div className="flex-1 border-t border-slate-100" />
                <span className="text-xs text-slate-300">{user.timestamp.replace('T', ' ').slice(0, 16)}</span>
              </div>

              {/* User question */}
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <User size={13} className="text-blue-600" />
                </div>
                <div className="flex-1">
                  <p className="text-xs text-slate-400 mb-1 font-medium">用户问题</p>
                  <div className="bg-blue-50 border border-blue-100 rounded-xl px-4 py-3 text-sm text-slate-800">
                    {user.content}
                  </div>
                </div>
              </div>

              {assistant && (
                <div className="flex items-start gap-3">
                  <div className="w-7 h-7 bg-teal-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Bot size={13} className="text-teal-600" />
                  </div>
                  <div className="flex-1 space-y-2">
                    {/* Rewritten query */}
                    {assistant.rewritten_query && (
                      <div className="flex items-center gap-2 text-xs text-slate-500 bg-slate-50 border border-slate-100 rounded-lg px-3 py-1.5">
                        <Search size={12} className="flex-shrink-0 text-slate-400" />
                        <span className="italic">改写查询：{assistant.rewritten_query}</span>
                      </div>
                    )}

                    {/* Sub-queries */}
                    {assistant.sub_queries && assistant.sub_queries.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {assistant.sub_queries.map((q, qi) => (
                          <span key={qi} className="px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-200 rounded text-xs">
                            {q}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Documents */}
                    {assistant.documents && assistant.documents.length > 0 && (
                      <div>
                        <p className="text-xs text-slate-400 mb-1 flex items-center gap-1">
                          <FileText size={11} />检索文档（{assistant.documents.length} 条）
                        </p>
                        <DocumentDrawer docs={assistant.documents} />
                      </div>
                    )}

                    {/* Answer */}
                    <div>
                      <p className="text-xs text-slate-400 mb-1 font-medium">系统回答</p>
                      <div className="bg-white border border-slate-100 shadow-sm rounded-xl px-4 py-3 text-sm text-slate-800 leading-relaxed">
                        <div className="prose prose-sm max-w-none">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{assistant.content}</ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
