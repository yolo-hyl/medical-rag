import { useState } from 'react';
import { BarChart3, Play, AlertCircle, Loader2 } from 'lucide-react';
import { apiEval } from '../api';
import type { EvalResult } from '../types';

function MetricCard({ label, value }: { label: string; value: number }) {
  const pct = value * 100;
  const color =
    value >= 0.7
      ? 'bg-green-50 border-green-200 text-green-700'
      : value >= 0.5
      ? 'bg-yellow-50 border-yellow-200 text-yellow-700'
      : 'bg-red-50 border-red-200 text-red-700';

  const barColor =
    value >= 0.7 ? 'bg-green-500' : value >= 0.5 ? 'bg-yellow-400' : 'bg-red-400';

  return (
    <div className={`p-4 border rounded-xl ${color}`}>
      <p className="text-xs font-medium opacity-80 mb-2">{label}</p>
      <p className="text-2xl font-bold mb-2">{pct.toFixed(1)}%</p>
      <div className="w-full bg-white bg-opacity-60 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all ${barColor}`}
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </div>
    </div>
  );
}

export default function EvalPage() {
  const [evalFile, setEvalFile] = useState('eval_data.json');
  const [sampleSize, setSampleSize] = useState('');
  const [queryField, setQueryField] = useState('');
  const [referenceField, setReferenceField] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvalResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleEval = async () => {
    if (!evalFile.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await apiEval(
        evalFile.trim(),
        sampleSize ? parseInt(sampleSize) : undefined,
        queryField.trim() || undefined,
        referenceField.trim() || undefined
      );
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : '评测失败');
    } finally {
      setLoading(false);
    }
  };

  const metrics = result?.metrics ?? {};
  const metricEntries = Object.entries(metrics).filter(
    ([, v]) => typeof v === 'number'
  ) as [string, number][];

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
          <BarChart3 size={22} className="text-blue-600" />
          RAG 评测
        </h2>
        <p className="text-sm text-slate-500 mt-1">对 RAG 系统进行自动化评测，分析检索与生成质量</p>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 space-y-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              评测数据文件路径
            </label>
            <input
              type="text"
              value={evalFile}
              onChange={(e) => setEvalFile(e.target.value)}
              placeholder="eval_data.json"
              className="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              样本数量 <span className="text-slate-400 font-normal">(可选)</span>
            </label>
            <input
              type="number"
              value={sampleSize}
              onChange={(e) => setSampleSize(e.target.value)}
              placeholder="全部样本"
              min={1}
              className="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              查询字段名 <span className="text-slate-400 font-normal">(可选)</span>
            </label>
            <input
              type="text"
              value={queryField}
              onChange={(e) => setQueryField(e.target.value)}
              placeholder="question"
              className="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              参考字段名 <span className="text-slate-400 font-normal">(可选)</span>
            </label>
            <input
              type="text"
              value={referenceField}
              onChange={(e) => setReferenceField(e.target.value)}
              placeholder="answer"
              className="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>

        <button
          onClick={handleEval}
          disabled={loading || !evalFile.trim()}
          className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
        >
          {loading ? (
            <Loader2 size={15} className="animate-spin" />
          ) : (
            <Play size={15} />
          )}
          {loading ? '评测运行中（可能需要数分钟）...' : '开始评测'}
        </button>

        {loading && (
          <div className="flex items-center gap-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <Loader2 size={16} className="text-blue-600 animate-spin flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-blue-700">评测正在进行中</p>
              <p className="text-xs text-blue-600 mt-0.5">
                RAG 评测需要为每个样本调用 LLM，可能需要较长时间，请耐心等待...
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
          <AlertCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-700">评测失败</p>
            <p className="text-xs text-red-600 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="mt-6 space-y-4">
          {metricEntries.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-3">评测指标</h3>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {metricEntries.map(([key, val]) => (
                  <MetricCard key={key} label={key} value={val} />
                ))}
              </div>
            </div>
          )}

          {result.summary && Object.keys(result.summary).length > 0 && (
            <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-100">
                <h3 className="text-sm font-semibold text-slate-700">评测摘要</h3>
              </div>
              <div className="divide-y divide-slate-100">
                {Object.entries(result.summary).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between px-4 py-2.5">
                    <span className="text-sm text-slate-600">{k}</span>
                    <span className="text-sm font-mono text-slate-700">{String(v)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-100">
              <h3 className="text-sm font-semibold text-slate-700">完整响应</h3>
            </div>
            <pre className="p-4 text-xs text-slate-600 overflow-x-auto scrollbar-thin font-mono leading-relaxed">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
