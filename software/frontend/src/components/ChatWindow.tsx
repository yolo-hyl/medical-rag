import { useEffect, useRef, useState } from 'react';
import { Send, Trash2, Bot, Plus } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage } from '../types';
import DocumentDrawer from './DocumentDrawer';

interface ChatWindowProps {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  isLoading: boolean;
  placeholder?: string;
  streaming?: boolean;
  onToggleStreaming?: (val: boolean) => void;
  sessionId?: string;
  onSessionChange?: (id: string) => void;
  showStreamToggle?: boolean;
  onClear?: () => void;
  onNewConversation?: () => void;
}


function MessageBubble({
  message,
  onClarificationClick,
}: {
  message: ChatMessage;
  onClarificationClick?: (q: string) => void;
}) {
  if (message.role === 'system') {
    const isProgress = message.progressStage !== undefined || message.content.startsWith('🔄') || message.content.includes('进度');
    return (
      <div className="flex justify-center my-2">
        <div
          className={`px-3 py-1.5 rounded-full text-xs font-medium border ${
            isProgress
              ? 'bg-amber-50 text-amber-700 border-amber-200'
              : 'bg-emerald-50 text-emerald-700 border-emerald-200'
          }`}
        >
          {message.content}
        </div>
      </div>
    );
  }

  if (message.role === 'user') {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[75%]">
          <div className="bg-blue-600 text-white px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed shadow-sm">
            {message.content}
          </div>
          <div className="text-xs text-slate-400 mt-1 text-right">
            {message.timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      </div>
    );
  }

  // assistant
  return (
    <div className="flex items-start gap-3 mb-4">
      <div className="w-8 h-8 bg-teal-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
        <Bot size={16} className="text-teal-600" />
      </div>
      <div className="max-w-[80%]">
        {/* 1. Streaming stages ABOVE content */}
        {message.streamingStages && message.streamingStages.length > 0 && (
          <div className="mb-1.5 space-y-1">
            {message.streamingStages.map((s, i) => (
              <div key={i} className="text-xs text-amber-600 bg-amber-50 border border-amber-100 rounded-full px-3 py-1 flex items-center gap-1.5 w-fit">
                <span className="animate-pulse">⏳</span> {s}
              </div>
            ))}
          </div>
        )}

        {/* 2. Rewritten query */}
        {message.rewrittenQuery && (
          <div className="mb-1.5 text-xs text-slate-400 bg-slate-50 border border-slate-100 rounded-lg px-3 py-1.5 flex items-center gap-1.5 w-fit">
            🔍 <span className="italic">改写：{message.rewrittenQuery}</span>
          </div>
        )}

        {/* 3. Background info */}
        {message.backgroundInfo && (
          <div className="mb-1.5 text-xs text-teal-700 bg-teal-50 border border-teal-100 rounded-lg px-3 py-1.5 flex items-start gap-1.5">
            👤 <span>背景：{message.backgroundInfo}</span>
          </div>
        )}

        {/* 4. Sub-queries */}
        {message.subQueries && message.subQueries.length > 0 && (
          <div className="mb-1.5 flex flex-wrap gap-1">
            {message.subQueries.map((q, i) => (
              <span
                key={i}
                className="px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-200 rounded text-xs"
              >
                {q}
              </span>
            ))}
          </div>
        )}

        {/* 5. Answer content */}
        <div className="bg-white border border-slate-100 shadow-sm px-4 py-3 rounded-2xl rounded-tl-sm text-sm text-slate-800 leading-relaxed">
          {message.isStreaming && !message.content ? (
            <span className="text-slate-400 italic">正在思考...</span>
          ) : (
            <div className="prose">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* 6. Clarification questions (fallback UI) */}
        {message.clarificationQuestions && message.clarificationQuestions.length > 0 && (
          <div className="mt-3">
            <p className="text-xs text-slate-500 mb-2">请补充以下信息：</p>
            <div className="space-y-1.5">
              {message.clarificationQuestions.map((q, i) => (
                <button
                  key={i}
                  onClick={() => onClarificationClick?.(q)}
                  className="block w-full text-left px-3 py-2 text-xs bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200 rounded-lg transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* 7. Documents */}
        {message.documents && message.documents.length > 0 && (
          <DocumentDrawer docs={message.documents} />
        )}

        <div className="text-xs text-slate-400 mt-1">
          {message.timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className="w-8 h-8 bg-teal-100 rounded-full flex items-center justify-center flex-shrink-0">
        <Bot size={16} className="text-teal-600" />
      </div>
      <div className="bg-white border border-slate-100 shadow-sm px-4 py-3 rounded-2xl rounded-tl-sm">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 bg-slate-400 rounded-full typing-dot" />
          <span className="w-2 h-2 bg-slate-400 rounded-full typing-dot" />
          <span className="w-2 h-2 bg-slate-400 rounded-full typing-dot" />
        </div>
      </div>
    </div>
  );
}

export default function ChatWindow({
  messages,
  onSend,
  isLoading,
  placeholder = '输入您的问题...',
  streaming = false,
  onToggleStreaming,
  sessionId = '',
  onSessionChange,
  showStreamToggle = false,
  onClear,
  onNewConversation,
}: ChatWindowProps) {
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isLoading) return;
    setInput('');
    onSend(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const showTyping = isLoading && !messages.some((m) => m.role === 'assistant' && m.isStreaming);

  return (
    <div className="flex flex-col h-full bg-slate-50">
      {/* Top bar */}
      {showStreamToggle && (
        <div className="flex items-center justify-between px-4 py-3 bg-white border-b border-slate-200 gap-4">
          <div className="flex items-center gap-3">
            {onSessionChange && (
              <div className="flex items-center gap-2">
                <label className="text-xs text-slate-500 whitespace-nowrap">会话 ID</label>
                <input
                  type="text"
                  value={sessionId}
                  onChange={(e) => onSessionChange(e.target.value)}
                  placeholder="自动生成"
                  className="border border-slate-200 rounded-md px-2 py-1 text-xs text-slate-700 w-36 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            )}
            {onToggleStreaming && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">流式输出</span>
                <button
                  onClick={() => onToggleStreaming(!streaming)}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none ${
                    streaming ? 'bg-blue-600' : 'bg-slate-300'
                  }`}
                >
                  <span
                    className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${
                      streaming ? 'translate-x-4' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            {onNewConversation && (
              <button
                onClick={onNewConversation}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-teal-600 hover:text-teal-700 hover:bg-teal-50 border border-teal-200 rounded-lg transition-colors"
              >
                <Plus size={13} />
                新对话
              </button>
            )}
            {onClear && (
              <button
                onClick={onClear}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                <Trash2 size={13} />
                清空
              </button>
            )}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 scrollbar-thin">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 bg-teal-100 rounded-full flex items-center justify-center mb-4">
              <Bot size={28} className="text-teal-600" />
            </div>
            <p className="text-slate-500 text-sm">开始对话吧</p>
            <p className="text-slate-400 text-xs mt-1">发送消息与医疗 AI 助手交流</p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            onClarificationClick={(q) => {
              setInput(q);
              textareaRef.current?.focus();
            }}
          />
        ))}

        {showTyping && <TypingIndicator />}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="px-4 py-4 bg-white border-t border-slate-200">
        <div className="flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            rows={2}
            disabled={isLoading}
            className="flex-1 border border-slate-200 rounded-xl px-4 py-3 text-sm text-slate-800 placeholder-slate-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-50 disabled:text-slate-400 scrollbar-thin"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="flex-shrink-0 w-10 h-10 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-xl flex items-center justify-center transition-colors shadow-sm"
            title="发送 (Ctrl+Enter)"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="text-xs text-slate-400 mt-1.5 text-right">Ctrl+Enter 发送</p>
      </div>
    </div>
  );
}
