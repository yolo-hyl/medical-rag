import { useState, useCallback } from 'react';
import { MessagesSquare } from 'lucide-react';
import { apiChat, apiChatStream } from '../api';
import ChatWindow from '../components/ChatWindow';
import type { AuthUser, ChatMessage, SSEEvent } from '../types';

let msgIdCounter = 0;
function newId() {
  return `msg-${Date.now()}-${++msgIdCounter}`;
}

function newUUID() {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

interface ChatPageProps {
  user: AuthUser;
}

export default function ChatPage({ user }: ChatPageProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState(() => `${user.user_id}-${newUUID()}`);
  const [streaming, setStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

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
          const res = await apiChat(text, sessionId || undefined);
          if (res.session_id) setSessionId(res.session_id);

          const botMsg: ChatMessage = {
            id: newId(),
            role: 'assistant',
            content: res.answer,
            documents: (res as Record<string, unknown>).sources as ChatMessage['documents'] ?? res.documents,
            rewrittenQuery: (res as Record<string, unknown>).rewritten_query as string | undefined,
            timestamp: new Date(),
          };
          appendMessage(botMsg);
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
          const stream = apiChatStream(text, sessionId || undefined);
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
        case 'documents': {
          const docs = Array.isArray(event.data) ? event.data : [];
          updateLastBotMessage((msg) => ({ ...msg, documents: [...(msg.documents ?? []), ...docs] }));
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
          break;
      }
    },
    [updateLastBotMessage]
  );

  const handleNewConversation = useCallback(() => {
    setMessages([]);
    setSessionId(`${user.user_id}-${newUUID()}`);
  }, [user.user_id]);

  const handleClear = useCallback(() => {
    setMessages([]);
    setSessionId(`${user.user_id}-${newUUID()}`);
  }, [user.user_id]);

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-4 border-b border-slate-200 bg-white flex-shrink-0">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <MessagesSquare size={22} className="text-blue-600" />
          多轮对话
        </h2>
        <p className="text-sm text-slate-500 mt-0.5">与医疗 AI 进行多轮上下文对话</p>
      </div>

      <div className="flex-1 overflow-hidden">
        <ChatWindow
          messages={messages}
          onSend={handleSend}
          isLoading={isLoading}
          placeholder="输入问题，开始多轮对话..."
          streaming={streaming}
          onToggleStreaming={setStreaming}
          sessionId={sessionId}
          onSessionChange={setSessionId}
          showStreamToggle={true}
          onClear={handleClear}
          onNewConversation={handleNewConversation}
        />
      </div>
    </div>
  );
}
