import { useState, useCallback } from 'react';
import { Brain } from 'lucide-react';
import { apiAgent, apiAgentStream } from '../api';
import ChatWindow from '../components/ChatWindow';
import ClarificationForm from '../components/ClarificationForm';
import type { AuthUser, ChatMessage, SSEEvent } from '../types';

let msgIdCounter = 0;
function newId() {
  return `agent-msg-${Date.now()}-${++msgIdCounter}`;
}

function newUUID() {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

interface AgentPageProps {
  user: AuthUser;
}

export default function AgentPage({ user }: AgentPageProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState(() => `${user.user_id}-agent-${newUUID()}`);
  const [streaming, setStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [clarificationQuestions, setClarificationQuestions] = useState<string[]>([]);

  const appendMessage = useCallback((msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const updateLastBotMessage = useCallback(
    (updater: (msg: ChatMessage) => ChatMessage) => {
      setMessages((prev) => {
        const idx = [...prev].reverse().findIndex((m) => m.role === 'assistant');
        if (idx === -1) return prev;
        const realIdx = prev.length - 1 - idx;
        const updated = [...prev];
        updated[realIdx] = updater(updated[realIdx]);
        return updated;
      });
    },
    []
  );

  const handleSend = useCallback(
    async (text: string) => {
      if (isLoading) return;

      const userMsg: ChatMessage = {
        id: newId(),
        role: 'user',
        content: text,
        timestamp: new Date(),
      };
      appendMessage(userMsg);
      setIsLoading(true);

      if (!streaming) {
        // Non-streaming
        try {
          const res = await apiAgent(text, sessionId || undefined);

          if (res.needs_clarification && res.clarification_questions?.length) {
            appendMessage({
              id: newId(),
              role: 'assistant',
              content: '为了更好地回答您的问题，请提供以下信息：',
              timestamp: new Date(),
            });
            setClarificationQuestions(res.clarification_questions ?? []);
          } else {
            const botMsg: ChatMessage = {
              id: newId(),
              role: 'assistant',
              content: res.answer,
              documents: res.documents,
              subQueries: res.sub_queries,
              timestamp: new Date(),
            };
            appendMessage(botMsg);
          }
        } catch (e) {
          const errMsg: ChatMessage = {
            id: newId(),
            role: 'assistant',
            content: `错误: ${e instanceof Error ? e.message : '请求失败'}`,
            timestamp: new Date(),
          };
          appendMessage(errMsg);
        } finally {
          setIsLoading(false);
        }
      } else {
        // Streaming
        appendMessage({
          id: newId(),
          role: 'assistant',
          content: '',
          isStreaming: true,
          streamingStages: [],
          timestamp: new Date(),
        });

        try {
          const stream = apiAgentStream(text, sessionId || undefined);
          for await (const event of stream) {
            handleStreamEvent(event);
          }
        } catch (e) {
          updateLastBotMessage((msg) => ({
            ...msg,
            content: `错误: ${e instanceof Error ? e.message : '流式请求失败'}`,
            isStreaming: false,
          }));
        } finally {
          updateLastBotMessage((msg) => ({ ...msg, isStreaming: false }));
          setIsLoading(false);
        }
      }
    },
    [isLoading, streaming, sessionId, appendMessage, updateLastBotMessage]
  );

  const handleStreamEvent = useCallback(
    (event: SSEEvent) => {
      switch (event.type) {
        case 'progress':
          updateLastBotMessage((msg) => ({
            ...msg,
            streamingStages: [...(msg.streamingStages ?? []), event.message ?? '处理中...'],
          }));
          break;
        case 'rewrite':
          updateLastBotMessage((msg) => ({
            ...msg,
            rewrittenQuery: String(event.data ?? ''),
          }));
          break;
        case 'background':
          updateLastBotMessage((msg) => ({
            ...msg,
            backgroundInfo: String(event.data ?? ''),
          }));
          break;
        case 'sub_queries': {
          const queries = Array.isArray(event.queries) ? event.queries : [];
          updateLastBotMessage((msg) => ({ ...msg, subQueries: queries }));
          break;
        }
        case 'documents': {
          const docs = Array.isArray(event.data) ? event.data : [];
          updateLastBotMessage((msg) => ({ ...msg, documents: [...(msg.documents ?? []), ...docs] }));
          break;
        }
        case 'clarification': {
          const questions = Array.isArray(event.questions) ? event.questions : [];
          updateLastBotMessage((msg) => ({
            ...msg,
            content: '为了更好地回答您的问题，请提供以下信息：',
            isStreaming: false,
          }));
          setClarificationQuestions(questions);
          setIsLoading(false);
          break;
        }
        case 'answer': {
          const answer = typeof event.data === 'string' ? event.data : (event.message ?? '');
          updateLastBotMessage((msg) => ({ ...msg, content: answer }));
          break;
        }
        case 'token': {
          const token = typeof event.data === 'string' ? event.data : (event.message ?? '');
          updateLastBotMessage((msg) => ({ ...msg, content: msg.content + token }));
          break;
        }
        case 'error':
          updateLastBotMessage((msg) => ({
            ...msg,
            content: `错误: ${event.message ?? '未知错误'}`,
            isStreaming: false,
          }));
          setIsLoading(false);
          break;
      }
    },
    [updateLastBotMessage]
  );

  const handleClarificationSubmit = useCallback((combined: string) => {
    setClarificationQuestions([]);
    // handleSend is stable and defined above via useCallback
    void handleSend(combined);
  }, [handleSend]);  // eslint-disable-line react-hooks/exhaustive-deps

  const handleNewConversation = useCallback(() => {
    setMessages([]);
    setSessionId(`${user.user_id}-agent-${newUUID()}`);
    setClarificationQuestions([]);
  }, [user.user_id]);

  const handleClear = useCallback(() => {
    setMessages([]);
    setSessionId(`${user.user_id}-agent-${newUUID()}`);
    setClarificationQuestions([]);
  }, [user.user_id]);

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-4 border-b border-slate-200 bg-white flex-shrink-0">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <Brain size={22} className="text-blue-600" />
          智能医疗 Agent
        </h2>
        <p className="text-sm text-slate-500 mt-0.5">
          基于 RAG 的智能医疗助手，支持多轮对话、澄清提问和复杂推理
        </p>
      </div>

      <div className="flex-1 overflow-hidden relative">
        <ChatWindow
          messages={messages}
          onSend={handleSend}
          isLoading={isLoading}
          placeholder="描述您的医疗问题，Agent 将自动分析并给出详细解答..."
          streaming={streaming}
          onToggleStreaming={setStreaming}
          sessionId={sessionId}
          onSessionChange={setSessionId}
          showStreamToggle={true}
          onClear={handleClear}
          onNewConversation={handleNewConversation}
        />
        {clarificationQuestions.length > 0 && (
          <ClarificationForm
            questions={clarificationQuestions}
            onSubmit={handleClarificationSubmit}
            onClose={() => setClarificationQuestions([])}
          />
        )}
      </div>
    </div>
  );
}
