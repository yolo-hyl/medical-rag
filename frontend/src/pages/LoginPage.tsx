import { useState } from 'react';
import { Stethoscope, AlertCircle } from 'lucide-react';
import { apiLogin, apiRegister } from '../api';
import type { AuthUser, UserRole } from '../types';

interface LoginPageProps {
  onLogin: (user: AuthUser) => void;
}

export default function LoginPage({ onLogin }: LoginPageProps) {
  const [tab, setTab] = useState<'login' | 'register'>('login');

  // Login state
  const [loginPhone, setLoginPhone] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loginLoading, setLoginLoading] = useState(false);

  // Register state
  const [regPhone, setRegPhone] = useState('');
  const [regPassword, setRegPassword] = useState('');
  const [regRole, setRegRole] = useState<UserRole>('patient');
  const [regError, setRegError] = useState<string | null>(null);
  const [regLoading, setRegLoading] = useState(false);

  const handleLogin = async () => {
    if (!loginPhone.trim() || !loginPassword.trim()) return;
    setLoginLoading(true);
    setLoginError(null);
    try {
      const res = await apiLogin(loginPhone.trim(), loginPassword);
      const user: AuthUser = {
        user_id: res.user_id,
        token: res.token,
        role: res.role as UserRole,
        phone: loginPhone.trim(),
      };
      localStorage.setItem('auth_user', JSON.stringify(user));
      onLogin(user);
    } catch (e) {
      setLoginError(e instanceof Error ? e.message : '登录失败');
    } finally {
      setLoginLoading(false);
    }
  };

  const handleRegister = async () => {
    if (!regPhone.trim() || regPhone.trim().length !== 11) {
      setRegError('手机号必须为11位');
      return;
    }
    if (regPassword.length < 6) {
      setRegError('密码至少6位');
      return;
    }
    setRegLoading(true);
    setRegError(null);
    try {
      await apiRegister(regPhone.trim(), regPassword, regRole);
      // Auto-login after register
      const res = await apiLogin(regPhone.trim(), regPassword);
      const user: AuthUser = {
        user_id: res.user_id,
        token: res.token,
        role: res.role as UserRole,
        phone: regPhone.trim(),
      };
      localStorage.setItem('auth_user', JSON.stringify(user));
      onLogin(user);
    } catch (e) {
      setRegError(e instanceof Error ? e.message : '注册失败');
    } finally {
      setRegLoading(false);
    }
  };

  const inputClass =
    'w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent';

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 to-slate-100 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-teal-600 rounded-2xl mb-4 shadow-lg">
            <Stethoscope size={28} className="text-white" />
          </div>
          <h1 className="text-2xl font-bold text-slate-800">医疗知识问答系统</h1>
          <p className="text-sm text-slate-500 mt-1">Medical RAG</p>
        </div>

        {/* Card */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-slate-200">
            {(['login', 'register'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`flex-1 py-3 text-sm font-medium transition-colors ${
                  tab === t
                    ? 'text-teal-700 border-b-2 border-teal-600 bg-teal-50/50'
                    : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                {t === 'login' ? '登录' : '注册'}
              </button>
            ))}
          </div>

          <div className="p-6 space-y-4">
            {tab === 'login' ? (
              <>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5">手机号</label>
                  <input
                    type="tel"
                    value={loginPhone}
                    onChange={(e) => setLoginPhone(e.target.value)}
                    placeholder="请输入11位手机号"
                    className={inputClass}
                    onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5">密码</label>
                  <input
                    type="password"
                    value={loginPassword}
                    onChange={(e) => setLoginPassword(e.target.value)}
                    placeholder="请输入密码"
                    className={inputClass}
                    onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                  />
                </div>
                {loginError && (
                  <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <AlertCircle size={15} className="text-red-500 flex-shrink-0" />
                    <p className="text-sm text-red-600">{loginError}</p>
                  </div>
                )}
                <button
                  onClick={handleLogin}
                  disabled={loginLoading || !loginPhone.trim() || !loginPassword.trim()}
                  className="w-full py-2.5 bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  {loginLoading ? '登录中...' : '登录'}
                </button>
              </>
            ) : (
              <>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5">手机号</label>
                  <input
                    type="tel"
                    value={regPhone}
                    onChange={(e) => setRegPhone(e.target.value)}
                    placeholder="请输入11位手机号"
                    maxLength={11}
                    className={inputClass}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5">密码</label>
                  <input
                    type="password"
                    value={regPassword}
                    onChange={(e) => setRegPassword(e.target.value)}
                    placeholder="至少6位"
                    className={inputClass}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">身份</label>
                  <div className="flex gap-3">
                    {(['patient', 'doctor', 'admin'] as UserRole[]).map((r) => (
                      <label
                        key={r}
                        className={`flex-1 flex items-center justify-center gap-1.5 py-2 border rounded-lg cursor-pointer text-sm transition-colors ${
                          regRole === r
                            ? 'border-teal-500 bg-teal-50 text-teal-700 font-medium'
                            : 'border-slate-200 text-slate-600 hover:border-teal-300'
                        }`}
                      >
                        <input
                          type="radio"
                          name="role"
                          value={r}
                          checked={regRole === r}
                          onChange={() => setRegRole(r)}
                          className="sr-only"
                        />
                        {r === 'patient' ? '患者' : r === 'doctor' ? '医生' : '管理员'}
                      </label>
                    ))}
                  </div>
                </div>
                {regError && (
                  <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <AlertCircle size={15} className="text-red-500 flex-shrink-0" />
                    <p className="text-sm text-red-600">{regError}</p>
                  </div>
                )}
                <button
                  onClick={handleRegister}
                  disabled={regLoading || !regPhone.trim() || !regPassword.trim()}
                  className="w-full py-2.5 bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  {regLoading ? '注册中...' : '注册并登录'}
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
