import { useState } from 'react';
import { Activity, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';
import { apiHealth } from '../api';
import type { HealthStatus } from '../types';

export default function HealthPage() {
  const [status, setStatus] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiHealth();
      setStatus(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : '健康检查失败');
    } finally {
      setLoading(false);
    }
  };

  const isHealthy = status?.status === 'ok' || status?.status === 'healthy';

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <Activity size={22} className="text-blue-600" />
          健康检查
        </h2>
        <p className="text-sm text-slate-500 mt-1">检查 API 服务及各组件的运行状态</p>
      </div>

      <button
        onClick={checkHealth}
        disabled={loading}
        className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
      >
        <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
        {loading ? '检查中...' : '检查健康状态'}
      </button>

      {error && (
        <div className="mt-4 flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
          <XCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-700">检查失败</p>
            <p className="text-xs text-red-600 mt-1">{error}</p>
          </div>
        </div>
      )}

      {status && (
        <div className="mt-6 space-y-4">
          {/* Overall status */}
          <div
            className={`flex items-center gap-4 p-4 rounded-xl border ${
              isHealthy
                ? 'bg-green-50 border-green-200'
                : 'bg-red-50 border-red-200'
            }`}
          >
            {isHealthy ? (
              <CheckCircle2 size={28} className="text-green-500 flex-shrink-0" />
            ) : (
              <XCircle size={28} className="text-red-500 flex-shrink-0" />
            )}
            <div>
              <p className={`font-semibold text-base ${isHealthy ? 'text-green-700' : 'text-red-700'}`}>
                {isHealthy ? '系统运行正常' : '系统存在异常'}
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                状态: <span className="font-mono font-medium">{String(status.status)}</span>
              </p>
            </div>
          </div>

          {/* Services breakdown */}
          {status.services && Object.keys(status.services).length > 0 && (
            <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-100">
                <h3 className="text-sm font-semibold text-slate-700">服务状态详情</h3>
              </div>
              <div className="divide-y divide-slate-100">
                {Object.entries(status.services).map(([key, val]) => {
                  const ok =
                    val === true ||
                    val === 'ok' ||
                    val === 'healthy' ||
                    val === 'connected' ||
                    (typeof val === 'number' && val > 0);
                  return (
                    <div key={key} className="flex items-center justify-between px-4 py-3">
                      <span className="text-sm text-slate-600">{key}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-500 font-mono">{String(val)}</span>
                        {ok ? (
                          <CheckCircle2 size={15} className="text-green-500" />
                        ) : (
                          <XCircle size={15} className="text-red-500" />
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Raw JSON */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-100">
              <h3 className="text-sm font-semibold text-slate-700">原始响应</h3>
            </div>
            <pre className="p-4 text-xs text-slate-600 overflow-x-auto scrollbar-thin font-mono leading-relaxed">
              {JSON.stringify(status, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
