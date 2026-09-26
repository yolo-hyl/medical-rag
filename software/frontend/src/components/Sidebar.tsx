import { useState } from 'react';
import {
  Activity,
  Database,
  Search,
  MessageSquare,
  MessagesSquare,
  BarChart3,
  Bot,
  Brain,
  BookOpen,
  FileSearch,
  ServerCog,
  Cpu,
  SlidersHorizontal,
  LogOut,
  ChevronDown,
  ChevronRight,
  Trash2,
  Clock,
} from 'lucide-react';
import type { AuthUser, SessionItem } from '../types';
import { apiGetHistory, apiDeleteSession } from '../api';
import HistoryDetailModal from './HistoryDetailModal';

interface NavItem {
  id: string;
  label: string;
  icon: string;
}

const PATIENT_NAV: NavItem[] = [
  { id: 'ask', label: '单轮问答', icon: 'MessageSquare' },
  { id: 'chat', label: '多轮对话', icon: 'MessagesSquare' },
  { id: 'agent', label: '智能医疗 Agent', icon: 'Brain' },
];

const DOCTOR_NAV: NavItem[] = [
  { id: 'search', label: '文档检索', icon: 'Search' },
  { id: 'ask', label: '单轮问答', icon: 'MessageSquare' },
  { id: 'chat', label: '多轮对话', icon: 'MessagesSquare' },
  { id: 'search-agent', label: '检索 Agent', icon: 'Bot' },
  { id: 'agent', label: '智能医疗 Agent', icon: 'Brain' },
  { id: 'literature', label: '文献检索', icon: 'BookOpen' },
  { id: 'case-search', label: '病例检索', icon: 'FileSearch' },
];

const ADMIN_NAV: NavItem[] = [
  { id: 'health', label: '健康检查', icon: 'Activity' },
  { id: 'ingest', label: '数据录入', icon: 'Database' },
  { id: 'data-manage', label: '数据管理', icon: 'ServerCog' },
  { id: 'model-switch', label: '模型切换', icon: 'Cpu' },
  { id: 'params-config', label: '参数配置', icon: 'SlidersHorizontal' },
];

function getNavItems(role: string): NavItem[] {
  if (role === 'admin') return ADMIN_NAV;
  if (role === 'doctor') return DOCTOR_NAV;
  return PATIENT_NAV;
}

function IconComponent({ name }: { name: string }) {
  const size = 18;
  switch (name) {
    case 'Activity': return <Activity size={size} />;
    case 'Database': return <Database size={size} />;
    case 'Search': return <Search size={size} />;
    case 'MessageSquare': return <MessageSquare size={size} />;
    case 'MessagesSquare': return <MessagesSquare size={size} />;
    case 'BarChart3': return <BarChart3 size={size} />;
    case 'Bot': return <Bot size={size} />;
    case 'Brain': return <Brain size={size} />;
    case 'BookOpen': return <BookOpen size={size} />;
    case 'FileSearch': return <FileSearch size={size} />;
    case 'ServerCog': return <ServerCog size={size} />;
    case 'Cpu': return <Cpu size={size} />;
    case 'SlidersHorizontal': return <SlidersHorizontal size={size} />;
    default: return <Activity size={size} />;
  }
}

interface HistoryPanelProps {
  serviceType: 'chat' | 'agent';
  label: string;
  token: string;
  onNavigate: (page: string) => void;
}

interface DetailState {
  sessionId: string;
  sessionTitle: string;
  serviceType: 'chat' | 'agent';
}

function HistoryPanel({ serviceType, label, token, onNavigate }: HistoryPanelProps) {
  const [open, setOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [detail, setDetail] = useState<DetailState | null>(null);

  const load = async () => {
    if (!open) {
      setLoading(true);
      try {
        const s = await apiGetHistory(serviceType, token);
        setSessions(s);
      } catch {
        setSessions([]);
      } finally {
        setLoading(false);
      }
    }
    setOpen((v) => !v);
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await apiDeleteSession(serviceType, id, token);
      setSessions((prev) => prev.filter((s) => s.id !== id));
    } catch {
      // ignore
    }
  };

  return (
    <>
      <div>
        <button
          onClick={load}
          className="w-full flex items-center justify-between px-3 py-2 text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors"
        >
          <span className="flex items-center gap-2">
            <Clock size={13} />
            {label}
          </span>
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </button>

        {open && (
          <div className="mt-1 space-y-0.5 pl-3">
            {loading && (
              <p className="text-xs text-slate-500 px-2 py-1">加载中...</p>
            )}
            {!loading && sessions.length === 0 && (
              <p className="text-xs text-slate-500 px-2 py-1">暂无历史</p>
            )}
            {sessions.map((s) => (
              <div
                key={s.id}
                className="group flex items-center gap-1 px-2 py-1.5 rounded-lg hover:bg-slate-800 cursor-pointer"
                onClick={() => setDetail({ sessionId: s.id, sessionTitle: s.title || '（无标题）', serviceType })}
              >
                <span className="flex-1 text-xs text-slate-400 group-hover:text-slate-200 truncate">
                  {s.title || '（无标题）'}
                </span>
                <button
                  onClick={(e) => handleDelete(e, s.id)}
                  className="opacity-0 group-hover:opacity-100 p-0.5 text-slate-500 hover:text-red-400 transition-colors"
                >
                  <Trash2 size={11} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {detail && (
        <HistoryDetailModal
          sessionId={detail.sessionId}
          sessionTitle={detail.sessionTitle}
          serviceType={detail.serviceType}
          token={token}
          onClose={() => setDetail(null)}
        />
      )}
    </>
  );
}

interface SidebarProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  user: AuthUser;
  onLogout: () => void;
}

export default function Sidebar({ currentPage, onNavigate, user, onLogout }: SidebarProps) {
  const navItems = getNavItems(user.role);
  const showHistory = user.role === 'patient' || user.role === 'doctor';

  return (
    <aside className="w-64 min-h-screen bg-slate-900 flex flex-col flex-shrink-0">
      {/* Logo */}
      <div className="px-6 py-5 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-teal-500 rounded-lg flex items-center justify-center flex-shrink-0">
            <span className="text-white text-sm font-bold">M</span>
          </div>
          <div>
            <h1 className="text-white font-semibold text-base leading-tight">Medical RAG</h1>
            <p className="text-slate-400 text-xs">智能医疗问答系统</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto scrollbar-thin">
        {navItems.map((item) => {
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 text-left ${
                isActive
                  ? 'bg-teal-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800'
              }`}
            >
              <span className={isActive ? 'text-white' : 'text-slate-500'}>
                <IconComponent name={item.icon} />
              </span>
              <span>{item.label}</span>
            </button>
          );
        })}

        {/* History panel for patient/doctor */}
        {showHistory && (
          <div className="pt-3 mt-3 border-t border-slate-700/50 space-y-1">
            <p className="px-3 text-xs text-slate-500 font-medium mb-1">历史对话</p>
            <HistoryPanel
              serviceType="chat"
              label="多轮对话"
              token={user.token}
              onNavigate={onNavigate}
            />
            <HistoryPanel
              serviceType="agent"
              label="智能 Agent"
              token={user.token}
              onNavigate={onNavigate}
            />
          </div>
        )}
      </nav>

      {/* Footer: user info + logout */}
      <div className="px-3 py-3 border-t border-slate-700">
        <div className="flex items-center justify-between px-3 py-2 rounded-lg">
          <div className="min-w-0">
            <p className="text-sm text-slate-300 font-medium truncate">{user.phone}</p>
            <p className="text-xs text-slate-500">
              {user.role === 'patient' ? '患者' : user.role === 'doctor' ? '医生' : '管理员'}
            </p>
          </div>
          <button
            onClick={onLogout}
            title="退出登录"
            className="p-1.5 text-slate-500 hover:text-red-400 hover:bg-slate-800 rounded-lg transition-colors flex-shrink-0"
          >
            <LogOut size={16} />
          </button>
        </div>
      </div>
    </aside>
  );
}
